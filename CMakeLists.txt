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

cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compilation commands")
endif()
project(tvm C CXX)
include(GNUInstallDirs)

# Declare native boolean switches and typed cache settings uniformly.
function(tvm_option variable type default description)
  if(type STREQUAL "BOOL")
    option(${variable} "${description}" "${default}")
  else()
    set(${variable} "${default}" CACHE ${type} "${description}")
  endif()
endfunction()

# ######### Common options ##########

tvm_option(TVM_USE_RPC BOOL ON "Build with RPC")
tvm_option(TVM_USE_CPP_RPC BOOL OFF "Build CPP RPC")
tvm_option(TVM_USE_IOS_RPC BOOL OFF "Build iOS RPC")
tvm_option(TVM_LOG_BEFORE_THROW BOOL OFF "Whether log before throw, for debugging purposes")
tvm_option(TVM_USE_MSVC_MT BOOL OFF "Build with MT")
tvm_option(TVM_HIDE_PRIVATE_SYMBOLS BOOL ON "Compile with -fvisibility=hidden.")
tvm_option(TVM_USE_GTEST STRING AUTO "Use GoogleTest for C++ sanity tests (OFF, ON, AUTO)")
tvm_option(TVM_USE_CCACHE STRING AUTO "Use ccache (OFF, ON, AUTO, or executable path)")
tvm_option(TVM_USE_ALTERNATIVE_LINKER STRING AUTO "Use 'mold' or 'lld' if found when invoking compiler to link artifact")

# ######### CPU libraries ##########

tvm_option(TVM_USE_BLAS STRING none "The blas library to be linked")
tvm_option(TVM_USE_MKL STRING OFF "Build with MKL (OFF, ON, or SDK path)")
tvm_option(TVM_USE_DNNL STRING OFF "Build with DNNL (OFF, ON, JSON, C_SRC, or SDK path)")
tvm_option(TVM_USE_AMX BOOL OFF "Enable Intel AMX")
tvm_option(TVM_USE_SORT BOOL ON "Build with sort support")
tvm_option(TVM_USE_RANDOM BOOL ON "Build with random support")

# ######### GPU libraries ##########

tvm_option(TVM_USE_NCCL STRING OFF "Build with NCCL (OFF, ON, or SDK path)")
tvm_option(TVM_USE_RCCL STRING OFF "Build with RCCL (OFF, ON, or SDK path)")
tvm_option(TVM_USE_NVSHMEM STRING OFF "Build with NVSHMEM (OFF, ON, or SDK path)")
tvm_option(TVM_USE_CUDNN STRING OFF "Build with cuDNN (OFF, ON, or SDK path)")
tvm_option(TVM_USE_CUDNN_FRONTEND STRING OFF "Build with cuDNN frontend (OFF, ON, or include root)")
tvm_option(TVM_USE_CUBLAS BOOL OFF "Build with cuBLAS")
tvm_option(TVM_USE_NVTX BOOL OFF "Build with NVTX")
tvm_option(TVM_USE_CUTLASS BOOL OFF "Build with CUTLASS")
tvm_option(TVM_USE_THRUST STRING OFF "Build with Thrust (OFF, ON, or CCCL path)")
tvm_option(TVM_USE_CURAND BOOL OFF "Build with cuRAND")
tvm_option(TVM_USE_HIPBLAS BOOL OFF "Build with ROCM:HIPBLAS")
tvm_option(TVM_USE_VLLM BOOL OFF "Build with vLLM support")

# ######### External code generators and runtimes ##########

tvm_option(TVM_USE_COREML BOOL OFF "Build with coreml support")
tvm_option(TVM_USE_TENSORRT_CODEGEN BOOL OFF "Build with TensorRT Codegen support")
tvm_option(TVM_USE_TENSORRT_RUNTIME STRING OFF "Build with TensorRT runtime (OFF, ON, or SDK path)")
tvm_option(TVM_USE_NNAPI_CODEGEN BOOL OFF "Build with NNAPI Codegen support")
tvm_option(TVM_USE_NNAPI_RUNTIME BOOL OFF "Build with NNAPI runtime")
tvm_option(TVM_USE_EXAMPLE_NPU_CODEGEN BOOL OFF "Build with Example NPU Codegen support")
tvm_option(TVM_USE_EXAMPLE_NPU_RUNTIME BOOL OFF "Build with Example NPU runtime")
tvm_option(TVM_USE_CLML STRING OFF "Build with CLML codegen (OFF, ON, or SDK path)")
tvm_option(TVM_USE_CLML_GRAPH_EXECUTOR STRING OFF "Build with CLML graph runtime (OFF, ON, or SDK path)")

set_property(CACHE TVM_USE_BLAS PROPERTY STRINGS none openblas atlas blas apple)
set_property(CACHE TVM_USE_GTEST PROPERTY STRINGS AUTO ON OFF)

# ######### Core dependencies ##########

set(_installed_tirx ON)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm-tirx/CMakeLists.txt")
  set(_installed_tirx OFF)
endif()
option(TVM_USE_INSTALLED_TIRX "Build against an installed apache-tvm-tirx package" ${_installed_tirx})
option(TVM_BUILD_PYTHON_MODULE "Build Python module with scikit-build-core" OFF)
if(TVM_BUILD_PYTHON_MODULE)
  set(CMAKE_INSTALL_LIBDIR lib)
endif()
if(TVM_USE_CLML_GRAPH_EXECUTOR)
  if(TVM_USE_CLML_GRAPH_EXECUTOR STREQUAL "ON")
    set(TVM_TIRX_USE_OPENCL "${CMAKE_CURRENT_SOURCE_DIR}/clml")
  else()
    set(TVM_TIRX_USE_OPENCL "${TVM_USE_CLML_GRAPH_EXECUTOR}")
  endif()
endif()
if(TVM_USE_INSTALLED_TIRX)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  if(NOT tvm_tirx_DIR)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_tirx.config --cmakedir
      OUTPUT_VARIABLE tvm_tirx_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
  endif()
  find_package(tvm_tirx CONFIG REQUIRED)
  if(NOT TARGET tvm_ffi::shared)
    if(NOT tvm_ffi_DIR)
      execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
        OUTPUT_VARIABLE tvm_ffi_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    endif()
    find_package(tvm_ffi CONFIG REQUIRED)
  endif()
  # Parent extensions use installed sidecars privately. They are not part of
  # the public TIRx package target interface.
  get_target_property(_tirx_runtime_location tvm_runtime::shared IMPORTED_LOCATION)
  get_filename_component(_tirx_libdir "${_tirx_runtime_location}" DIRECTORY)
  foreach(_backend cuda rocm opencl vulkan metal)
    find_library(_tirx_backend_library NAMES tvm_runtime_${_backend}
      PATHS "${_tirx_libdir}" NO_DEFAULT_PATH NO_CACHE)
    if(_tirx_backend_library AND NOT TARGET tvm_runtime_${_backend})
      add_library(tvm_runtime_${_backend} UNKNOWN IMPORTED)
      set_target_properties(tvm_runtime_${_backend} PROPERTIES
        IMPORTED_LOCATION "${_tirx_backend_library}")
      target_link_libraries(tvm_runtime_${_backend} INTERFACE tvm_runtime::shared tvm_ffi::shared)
    endif()
    unset(_tirx_backend_library)
  endforeach()
  foreach(_backend CUDA ROCM)
    # Installed packages expose libraries, not cached build-option state.
    if(NOT DEFINED TVM_TIRX_USE_${_backend} OR "${TVM_TIRX_USE_${_backend}}" MATCHES "^[Aa][Uu][Tt][Oo]$")
      string(TOLOWER "${_backend}" _backend_lower)
      set(TVM_TIRX_USE_${_backend} OFF)
      if(TARGET tvm_runtime_${_backend_lower})
        set(TVM_TIRX_USE_${_backend} ON)
      endif()
    endif()
  endforeach()
  if(TARGET tvm_runtime_opencl)
    if(WIN32)
      target_compile_definitions(tvm_runtime_opencl INTERFACE "TVM_OPENCL_DLL=__declspec(dllimport)")
    else()
      target_compile_definitions(tvm_runtime_opencl INTERFACE
        "TVM_OPENCL_DLL=__attribute__((visibility(\"default\")))")
    endif()
  endif()
  if(NOT TVM_TIRX_VERSION STREQUAL "0.1.0a4")
    message(FATAL_ERROR "TVM requires apache-tvm-tirx 0.1.0a4")
  endif()
else()
  # Keep source-built core libraries beside the parent libraries.
  set(_tirx_output_defaults "")
  foreach(_kind ARCHIVE LIBRARY RUNTIME)
    if(NOT DEFINED CMAKE_${_kind}_OUTPUT_DIRECTORY)
      set(CMAKE_${_kind}_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
      list(APPEND _tirx_output_defaults ${_kind})
    endif()
  endforeach()
  add_subdirectory(3rdparty/tvm-tirx)
  if(TARGET tirx_cpptest)
    set_target_properties(tirx_cpptest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
      string(TOUPPER "${_config}" _config_upper)
      set_target_properties(tirx_cpptest PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${_config_upper} "${CMAKE_BINARY_DIR}")
    endforeach()
  endif()
  foreach(_kind IN LISTS _tirx_output_defaults)
    unset(CMAKE_${_kind}_OUTPUT_DIRECTORY)
  endforeach()
endif()
# ######### Parent build settings ##########

include(cmake/utils/Utils.cmake)
include(cmake/utils/Linker.cmake)
include(cmake/utils/Library.cmake)
include(cmake/utils/CCache.cmake)
include("${TVM_TIRX_CMAKE_DIR}/utils/FindCUDA.cmake")
include("${TVM_TIRX_CMAKE_DIR}/utils/FindROCM.cmake")
include("${TVM_TIRX_CMAKE_DIR}/utils/FindOpenCL.cmake")
include(cmake/utils/FindNCCL.cmake)
include(cmake/utils/FindRCCL.cmake)
include(cmake/utils/FindNVSHMEM.cmake)
if(NOT TARGET tvm_ffi_header)
  add_library(tvm_ffi_header ALIAS tvm_ffi::header)
endif()
# Existing parent integration modules use the unqualified runtime target.
if(NOT TARGET tvm_runtime)
  add_library(tvm_runtime ALIAS tvm_runtime::shared)
endif()
set(BUILD_FOR_ANDROID "${ANDROID}")
if(DEFINED SKBUILD_PROJECT_VERSION_FULL AND NOT DEFINED TVM_VERSION)
  set(TVM_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
elseif(NOT DEFINED TVM_VERSION)
  set(TVM_VERSION "0.28.dev0")
endif()
find_and_set_linker("${TVM_USE_ALTERNATIVE_LINKER}")

# ######### Compiler and runtime extensions ##########

add_library(tvm_compiler SHARED)
tvm_configure_target(tvm_compiler)
set(TVM_RUNTIME_BACKEND_LIBS "")
file(GLOB_RECURSE COMPILER_SRCS CONFIGURE_DEPENDS
    src/te/*.cc
    src/s_tir/*.cc
    src/topi/*.cc
    src/driver/*.cc
    src/relax/ir/*.cc
    src/relax/op/*.cc
    src/relax/analysis/*.cc
    src/relax/transform/*.cc
    src/relax/backend/vm/*.cc
    src/relax/backend/adreno/*.cc
    src/relax/backend/task_extraction.cc
    src/relax/backend/pattern_registry.cc
    src/relax/utils.cc
    src/relax/distributed/*.cc
    src/relax/distributed/transform/*.cc
    src/relax/op/distributed/*.cc
    src/relax/script/*.cc
    src/relax/testing/*.cc
)

# Timing, VM, RPC, Disco and optional integrations remain parent runtime extensions.
add_library(tvm_runtime_extra SHARED)
tvm_configure_target(tvm_runtime_extra)
add_library(tvm_runtime_extra_defs INTERFACE)
target_link_libraries(tvm_runtime_extra_defs INTERFACE
  tvm_runtime::shared tvm_ffi::shared)
target_include_directories(tvm_runtime_extra_defs SYSTEM INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm-tirx/3rdparty/compiler-rt")
target_compile_definitions(tvm_runtime_extra_defs INTERFACE TVM_RUNTIME_EXTRA_EXPORTS TVM_FFI_EXPORTS)
target_compile_features(tvm_runtime_extra_defs INTERFACE cxx_std_20)
target_link_libraries(tvm_runtime_extra PRIVATE tvm_runtime_extra_defs)
file(GLOB _vm_sources CONFIGURE_DEPENDS src/runtime/vm/*.cc src/runtime/rpc/minrpc/*.cc)
target_sources(tvm_runtime_extra PRIVATE src/runtime/timer.cc ${_vm_sources})
if(TVM_USE_RPC)
  file(GLOB _rpc_sources CONFIGURE_DEPENDS src/runtime/rpc/*.cc)
  target_sources(tvm_runtime_extra PRIVATE ${_rpc_sources})
  if(WIN32)
    target_link_libraries(tvm_runtime_extra PRIVATE ws2_32)
  endif()
endif()
if(TVM_TIRX_USE_CUDA)
  if(NOT TARGET tvm_runtime_cuda)
    message(FATAL_ERROR "Parent CUDA support requires a CUDA-enabled core")
  endif()
  # The frontend also compiles and links against the cuDNN library.
  if(TVM_USE_CUDNN_FRONTEND AND NOT TVM_USE_CUDNN)
    find_cuda(${TVM_TIRX_USE_CUDA} ON)
  else()
    find_cuda(${TVM_TIRX_USE_CUDA} ${TVM_USE_CUDNN})
  endif()
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  target_include_directories(tvm_runtime_extra_defs SYSTEM INTERFACE ${CUDA_INCLUDE_DIRS})
  target_include_directories(tvm_compiler SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
  file(GLOB _vm_cuda_sources CONFIGURE_DEPENDS src/runtime/vm/cuda/*.cc)
  target_sources(tvm_runtime_extra PRIVATE ${_vm_cuda_sources})
endif()
if(TVM_TIRX_USE_ROCM)
  if(NOT TARGET tvm_runtime_rocm)
    message(FATAL_ERROR "Parent ROCm support requires a ROCm-enabled core")
  endif()
  if("${TVM_TIRX_USE_ROCM}" MATCHES ${IS_TRUE_PATTERN} AND IS_DIRECTORY "${TVM_TIRX_ROCM_PATH}")
    find_rocm("${TVM_TIRX_ROCM_PATH}")
  else()
    find_rocm(${TVM_TIRX_USE_ROCM})
  endif()
  target_include_directories(tvm_runtime_extra_defs SYSTEM INTERFACE ${ROCM_INCLUDE_DIRS})
  target_compile_definitions(tvm_runtime_extra_defs INTERFACE __HIP_PLATFORM_HCC__=1 __HIP_PLATFORM_AMD__=1)
  target_include_directories(tvm_compiler SYSTEM PRIVATE ${ROCM_INCLUDE_DIRS})
  target_compile_definitions(tvm_compiler PRIVATE __HIP_PLATFORM_HCC__=1 __HIP_PLATFORM_AMD__=1)
endif()
if(TVM_USE_NVTX)
  target_compile_definitions(tvm_runtime_extra_defs INTERFACE TVM_NVTX_ENABLED=1)
endif()
include(cmake/modules/contrib/CUDA.cmake)
include(cmake/modules/contrib/ROCM.cmake)
include(cmake/modules/contrib/BLAS.cmake)
include(cmake/modules/contrib/DNNL.cmake)
include(cmake/modules/contrib/AMX.cmake)
include(cmake/modules/contrib/CUTLASS.cmake)
include(cmake/modules/contrib/CLML.cmake)
include(cmake/modules/contrib/Random.cmake)
include(cmake/modules/contrib/Sort.cmake)
include(cmake/modules/contrib/CoreML.cmake)
include(cmake/modules/contrib/TensorRT.cmake)
include(cmake/modules/contrib/NNAPI.cmake)
include(cmake/modules/contrib/ExampleNPU.cmake)
include(cmake/modules/contrib/vllm.cmake)
if(TVM_USE_CLML_GRAPH_EXECUTOR)
  target_link_libraries(tvm_clml_objs PRIVATE tvm_runtime_opencl)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_runtime_opencl)
endif()
if(TARGET tvm_clml_objs)
  # CLML shares OpenCL implementation types; these are source inputs, not an
  # installed core interface. The parent sdist carries their bounded closure.
  target_include_directories(tvm_clml_objs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm-tirx/src")
endif()
# Retain the upstream runtime-extra assembly module and its target composition.
include(cmake/modules/RuntimeExtra.cmake)
target_link_libraries(tvm_runtime_extra PRIVATE ${CMAKE_DL_LIBS})
target_sources(tvm_compiler PRIVATE ${COMPILER_SRCS})
target_compile_definitions(tvm_compiler PRIVATE TVM_EXPORTS TVM_FFI_EXPORTS)
target_link_libraries(tvm_compiler PUBLIC tvm_tirx_compiler::shared tvm_ffi::shared tvm_runtime_extra)
target_include_directories(tvm_compiler SYSTEM PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm-tirx/3rdparty/compiler-rt")
foreach(_target tvm_compiler tvm_runtime_extra)
  target_include_directories(${_target} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  if(APPLE)
    target_link_options(${_target} PRIVATE "LINKER:-undefined,error")
  elseif(NOT MSVC)
    target_link_options(${_target} PRIVATE "LINKER:--no-undefined")
  endif()
  set_target_properties(${_target} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  if(MSVC)
    foreach(_config RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
      set_target_properties(${_target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_config} "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY_${_config} "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY_${_config} "${CMAKE_BINARY_DIR}/lib")
    endforeach()
  endif()
  tvm_configure_target_library(${_target})
  if(APPLE)
    set_property(TARGET ${_target} APPEND PROPERTY INSTALL_RPATH "@loader_path/../../tvm_tirx/lib")
  elseif(UNIX)
    set_property(TARGET ${_target} APPEND PROPERTY INSTALL_RPATH "\$ORIGIN/../../tvm_tirx/lib")
  endif()
  if(TVM_BUILD_PYTHON_MODULE)
    if(APPLE)
      set_property(TARGET ${_target} APPEND PROPERTY INSTALL_RPATH "@loader_path/../../tvm_ffi/lib")
    elseif(UNIX)
      set_property(TARGET ${_target} APPEND PROPERTY INSTALL_RPATH "\$ORIGIN/../../tvm_ffi/lib")
    endif()
  endif()
  tvm_ffi_add_apple_dsymutil(${_target})
endforeach()
add_subdirectory(tests/cpp)
if(GTEST_FOUND)
  enable_testing()
  include(CTest)
endif()
if(TARGET cpptest AND TARGET tirx_cpptest)
  add_dependencies(cpptest tirx_cpptest)
endif()
add_custom_target(runtime DEPENDS tvm_runtime_extra)
add_dependencies(runtime tvm_runtime::shared)
set(TVM_RUNTIME_BACKEND_LIBS tvm_runtime_extra)
foreach(_backend cuda opencl vulkan metal rocm)
  if(TARGET tvm_runtime_${_backend})
    list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_${_backend})
  endif()
endforeach()
if(TVM_USE_CPP_RPC)
  add_subdirectory(apps/cpp_rpc)
endif()
if(TVM_USE_IOS_RPC)
  add_subdirectory(apps/ios_rpc)
endif()

# ######### Installation (parent package only) ##########

install(TARGETS tvm_compiler tvm_runtime_extra
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm)
if(TVM_TIRX_USE_CUDA AND TVM_USE_CUTLASS)
  foreach(_target fpA_intB_gemm flash_attn)
    tvm_configure_target_library(${_target})
  endforeach()
  install(TARGETS fpA_intB_gemm flash_attn
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm)
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT tvm FILES_MATCHING PATTERN "*.h")
# Parent minRPC compilation retains its own source and private header closure.
install(DIRECTORY src/runtime/rpc/ DESTINATION src/runtime/rpc
  COMPONENT tvm FILES_MATCHING PATTERN "*.h" PATTERN "*.cc")
install(DIRECTORY src/support/ DESTINATION src/support
  COMPONENT tvm FILES_MATCHING PATTERN "*.h")
install(DIRECTORY src/runtime/extra/ DESTINATION src/runtime/extra
  COMPONENT tvm FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")
if(TVM_BUILD_PYTHON_MODULE)
  install(DIRECTORY web/ DESTINATION web COMPONENT tvm)
  install(DIRECTORY licenses/ DESTINATION licenses COMPONENT tvm)
  install(FILES LICENSE NOTICE DESTINATION . COMPONENT tvm)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include")
    install(DIRECTORY 3rdparty/cutlass/include/ DESTINATION 3rdparty/cutlass/include
      COMPONENT tvm FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
  endif()
endif()
