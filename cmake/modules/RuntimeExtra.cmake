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

# Assemble the runtime-extra target declared before the backend modules.
# Disco core sources.
file(GLOB _disco_core_srcs CONFIGURE_DEPENDS src/runtime/extra/disco/*.cc)
add_library(tvm_disco_objs OBJECT ${_disco_core_srcs})
tvm_configure_target(tvm_disco_objs)
target_link_libraries(tvm_disco_objs PRIVATE tvm_runtime_extra_defs)
target_link_libraries(tvm_runtime_extra PRIVATE tvm_disco_objs)

# Distributed disco.
file(GLOB _disco_dist_srcs CONFIGURE_DEPENDS src/runtime/extra/disco/distributed/*.cc)
add_library(tvm_disco_distributed_objs OBJECT ${_disco_dist_srcs})
tvm_configure_target(tvm_disco_distributed_objs)
target_link_libraries(tvm_disco_distributed_objs PRIVATE tvm_runtime_extra_defs)
target_link_libraries(tvm_runtime_extra PRIVATE tvm_disco_distributed_objs)

# NCCL / cuda_ipc — requires CUDA + NCCL.
if(TVM_TIRX_USE_CUDA AND TVM_USE_NCCL)
  find_nccl(${TVM_USE_NCCL})
  file(GLOB _nccl_srcs CONFIGURE_DEPENDS src/runtime/extra/disco/nccl/*.cc src/runtime/extra/disco/cuda_ipc/*.cc 3rdparty/tensorrt_llm/*.cu)
  add_library(tvm_nccl_objs OBJECT ${_nccl_srcs})
  tvm_configure_target(tvm_nccl_objs)
  target_compile_definitions(tvm_nccl_objs PRIVATE TVM_NCCL_RCCL_SWITCH=0)
  target_link_libraries(tvm_nccl_objs PRIVATE tvm_runtime_extra_defs nccl)
  find_library(LIBRT rt)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_nccl_objs nccl ${LIBRT})
endif()

# NVSHMEM.
if(TVM_TIRX_USE_CUDA AND TVM_USE_NVSHMEM)
  find_nvshmem(${TVM_USE_NVSHMEM})
  if(NOT NVSHMEM_FOUND)
    message(FATAL_ERROR "Cannot find NVSHMEM, TVM_USE_NVSHMEM=" ${TVM_USE_NVSHMEM})
  endif()
  file(GLOB _nvshmem_srcs CONFIGURE_DEPENDS src/runtime/extra/contrib/nvshmem/*.cc src/runtime/extra/contrib/nvshmem/*.cu)
  add_library(tvm_nvshmem_objs OBJECT ${_nvshmem_srcs})
  tvm_configure_target(tvm_nvshmem_objs)
  set_target_properties(tvm_nvshmem_objs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(tvm_nvshmem_objs PRIVATE tvm_runtime_extra_defs)
  target_include_directories(tvm_nvshmem_objs PUBLIC ${NVSHMEM_INCLUDE_DIR})
  # CUDA 13 moved libcu++ under include/cccl. NVSHMEM's headers include
  # <cuda/std/...>, which nvcc resolves on its own -- the host compiler
  # building this target's .cc sources does not, so name the directory.
  if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
    target_include_directories(tvm_nvshmem_objs SYSTEM PUBLIC
                               "${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
  endif()
  find_library(NVSHMEM_HOST nvshmem_host ${NVSHMEM_LIB_DIR})
  find_library(NVSHMEM_DEVICE nvshmem_device ${NVSHMEM_LIB_DIR})
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_nvshmem_objs ${NVSHMEM_HOST} ${NVSHMEM_DEVICE})
  set_target_properties(tvm_runtime_extra PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# RCCL.
if(TVM_TIRX_USE_ROCM AND TVM_USE_RCCL)
  find_rccl(${TVM_USE_RCCL})
  file(GLOB _rccl_srcs CONFIGURE_DEPENDS src/runtime/extra/disco/nccl/*.cc)
  add_library(tvm_rccl_objs OBJECT ${_rccl_srcs})
  tvm_configure_target(tvm_rccl_objs)
  target_compile_definitions(tvm_rccl_objs PRIVATE TVM_NCCL_RCCL_SWITCH=1)
  target_link_libraries(tvm_rccl_objs PRIVATE tvm_runtime_extra_defs rccl)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_rccl_objs rccl)
endif()

target_link_libraries(tvm_runtime_extra PUBLIC tvm_runtime)

# If disco/cuda_ipc is included, link the CUDA DSO.
if(TVM_TIRX_USE_CUDA)
  target_link_libraries(tvm_runtime_extra PUBLIC tvm_runtime_cuda)
  target_link_libraries(tvm_runtime_extra PRIVATE ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY})
endif()

if(TVM_TIRX_USE_ROCM)
  target_link_libraries(tvm_runtime_extra PRIVATE ${ROCM_HIPHCC_LIBRARY})
endif()

# CUTLASS fpA_intB_gemm and flash_attn are separate shared libs.
if(TVM_TIRX_USE_CUDA AND TVM_USE_CUTLASS)
  target_link_libraries(tvm_runtime_extra PRIVATE fpA_intB_gemm fpA_intB_gemm_tvm)
  target_link_libraries(tvm_runtime_extra PRIVATE -Wl,--no-as-needed flash_attn)
endif()
