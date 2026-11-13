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

if(TVM_TIRX_USE_CUDA AND TVM_USE_CUTLASS)
  file(GLOB CUTLASS_CONTRIB_SRC CONFIGURE_DEPENDS src/relax/backend/contrib/cutlass/*.cc
  )
  list(APPEND COMPILER_SRCS ${CUTLASS_CONTRIB_SRC})

  set(FPA_INTB_GEMM_TVM_BINDING ON)
  set(FPA_INTB_GEMM_TVM_HOME ${PROJECT_SOURCE_DIR})

  ### Build cutlass runtime objects for fpA_intB_gemm using its cutlass submodule
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm)
  tvm_configure_target(fpA_intB_gemm KEEP_SYMBOLS)
  tvm_configure_target(fpA_intB_gemm_tvm)
  target_include_directories(fpA_intB_gemm PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm/cutlass/include
  )
  target_link_libraries(fpA_intB_gemm_tvm PRIVATE tvm_runtime_extra_defs)

  set(CUTLASS_FPA_INTB_RUNTIME_SRCS "")
  list(APPEND CUTLASS_FPA_INTB_RUNTIME_SRCS src/runtime/extra/contrib/cutlass/weight_preprocess.cc)
  add_library(fpA_intB_cutlass_objs OBJECT ${CUTLASS_FPA_INTB_RUNTIME_SRCS})
  tvm_configure_target(fpA_intB_cutlass_objs)
  target_link_libraries(fpA_intB_cutlass_objs PRIVATE tvm_runtime_extra_defs)
  target_include_directories(fpA_intB_cutlass_objs PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm/cutlass/include
  )
  target_link_libraries(tvm_runtime_extra PRIVATE fpA_intB_cutlass_objs)

  ### Build cutlass runtime objects for flash attention
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/libflash_attn)
  tvm_configure_target(flash_attn KEEP_SYMBOLS)
  target_include_directories(flash_attn PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/libflash_attn
    ${PROJECT_SOURCE_DIR}/3rdparty/libflash_attn/cutlass/include
  )

  ### Build cutlass runtime objects using TVM's 3rdparty/cutlass submodule
  set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass)
  set(TVM_CUTLASS_RUNTIME_SRCS "")

  if(CMAKE_CUDA_ARCHITECTURES MATCHES "90a")
    list(APPEND TVM_CUTLASS_RUNTIME_SRCS src/runtime/extra/contrib/cutlass/fp16_group_gemm_sm90.cu)
    list(APPEND TVM_CUTLASS_RUNTIME_SRCS src/runtime/extra/contrib/cutlass/fp8_group_gemm_sm90.cu)
    list(APPEND TVM_CUTLASS_RUNTIME_SRCS src/runtime/extra/contrib/cutlass/fp8_gemm.cu)
    list(APPEND TVM_CUTLASS_RUNTIME_SRCS src/runtime/extra/contrib/cutlass/fp8_groupwise_scaled_gemm_sm90.cu)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "100a")
    list(APPEND TVM_CUTLASS_RUNTIME_SRCS src/runtime/extra/contrib/cutlass/fp16_group_gemm_sm100.cu)
    list(APPEND TVM_CUTLASS_RUNTIME_SRCS src/runtime/extra/contrib/cutlass/fp8_groupwise_scaled_gemm_sm100.cu)
    list(APPEND TVM_CUTLASS_RUNTIME_SRCS src/runtime/extra/contrib/cutlass/fp8_groupwise_scaled_group_gemm_sm100.cu)
  endif()
  if(TVM_CUTLASS_RUNTIME_SRCS)
    add_library(tvm_cutlass_objs OBJECT ${TVM_CUTLASS_RUNTIME_SRCS})
    tvm_configure_target(tvm_cutlass_objs)
    target_compile_options(tvm_cutlass_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
    target_include_directories(tvm_cutlass_objs PRIVATE
      ${CUTLASS_DIR}/include
      ${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm/cutlass_extensions/include
    )
    target_link_libraries(tvm_cutlass_objs PRIVATE tvm_runtime_extra_defs)
    # Note: enable this to get more detailed logs for cutlass kernels
    # target_compile_definitions(tvm_cutlass_objs PRIVATE CUTLASS_DEBUG_TRACE_LEVEL=2)
    target_link_libraries(tvm_runtime_extra PRIVATE tvm_cutlass_objs)
  endif()

  message(STATUS "Build with CUTLASS")
endif()
