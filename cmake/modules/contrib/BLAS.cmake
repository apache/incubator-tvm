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

if(TVM_USE_BLAS STREQUAL "openblas")
  find_library(BLAS_LIBRARY openblas)
  add_library(tvm_blas_objs OBJECT src/runtime/extra/contrib/cblas/cblas.cc)
  tvm_configure_target(tvm_blas_objs)
  target_link_libraries(tvm_blas_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_blas_objs ${BLAS_LIBRARY})
  message(STATUS "Using BLAS library " ${BLAS_LIBRARY})
  find_path(BLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
  if(BLAS_INCLUDE_DIR)
    message(STATUS "Using BLAS header in " ${BLAS_INCLUDE_DIR})
    target_include_directories(tvm_blas_objs SYSTEM PRIVATE ${BLAS_INCLUDE_DIR})
  endif()
elseif(TVM_USE_BLAS STREQUAL "atlas" OR TVM_USE_BLAS STREQUAL "blas")
  find_library(BLAS_LIBRARY cblas)
  add_library(tvm_blas_objs OBJECT src/runtime/extra/contrib/cblas/cblas.cc)
  tvm_configure_target(tvm_blas_objs)
  target_link_libraries(tvm_blas_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_blas_objs ${BLAS_LIBRARY})
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(TVM_USE_BLAS STREQUAL "apple")
  find_library(BLAS_LIBRARY Accelerate)
  add_library(tvm_blas_objs OBJECT src/runtime/extra/contrib/cblas/cblas.cc)
  target_include_directories(tvm_blas_objs SYSTEM PRIVATE
    ${BLAS_LIBRARY}/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/)
  tvm_configure_target(tvm_blas_objs)
  target_link_libraries(tvm_blas_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_blas_objs ${BLAS_LIBRARY})
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(TVM_USE_BLAS STREQUAL "mkl")
  message(FATAL_ERROR "TVM_USE_BLAS=mkl has been removed. Use TVM_USE_MKL=ON instead.")
elseif(TVM_USE_BLAS STREQUAL "none")
  # pass
else()
  message(FATAL_ERROR "Invalid option: TVM_USE_BLAS=" ${TVM_USE_BLAS})
endif()

if(TVM_USE_MKL)
  if(NOT IS_DIRECTORY ${TVM_USE_MKL})
    set(TVM_USE_MKL /opt/intel/mkl)
  endif()
  if(APPLE)
    find_library(BLAS_LIBRARY_MKL NAMES mklml mkl_rt HINTS ${TVM_USE_MKL}/lib/ ${TVM_USE_MKL}/lib/intel64)
  elseif(UNIX)
    find_library(BLAS_LIBRARY_MKL NAMES mkl_rt mklml_gnu HINTS ${TVM_USE_MKL}/lib/ ${TVM_USE_MKL}/lib/intel64)
  elseif(MSVC)
    find_library(BLAS_LIBRARY_MKL NAMES mkl_rt HINTS ${TVM_USE_MKL}/lib/ ${TVM_USE_MKL}/lib/intel64_win)
  endif()
  add_library(tvm_mkl_objs OBJECT src/runtime/extra/contrib/cblas/mkl.cc)
  tvm_configure_target(tvm_mkl_objs)
  target_include_directories(tvm_mkl_objs SYSTEM PRIVATE ${TVM_USE_MKL}/include)
  target_link_libraries(tvm_mkl_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_mkl_objs ${BLAS_LIBRARY_MKL})
  target_compile_definitions(tvm_mkl_objs PRIVATE USE_MKL_BLAS=1)
  message(STATUS "Use MKL library " ${BLAS_LIBRARY_MKL})
endif()
