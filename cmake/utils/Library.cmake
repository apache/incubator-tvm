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

# Apply parent compilation policy to the targets that compile, including objects.
# KEEP_SYMBOLS preserves the unannotated public APIs of bundled third-party DSOs.
function(tvm_configure_target target)
  set_target_properties(${target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(${target} PRIVATE TVM_VERSION="${TVM_VERSION}")
  if(TVM_LOG_BEFORE_THROW)
    target_compile_definitions(${target} PRIVATE TVM_LOG_BEFORE_THROW=1)
  endif()
  if(MSVC)
    target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
      _SCL_SECURE_NO_WARNINGS _ENABLE_EXTENDED_ALIGNED_STORAGE NOMINMAX)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj;/permissive-;/EHsc>)
    if(TVM_USE_MSVC_MT)
      set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>)
    if(TVM_HIDE_PRIVATE_SYMBOLS AND NOT ARGV1 STREQUAL "KEEP_SYMBOLS")
      set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET hidden C_VISIBILITY_PRESET hidden)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>)
    endif()
  endif()
  target_link_options(${target} PRIVATE ${TVM_LINKER_FLAGS})
endfunction()

# Keep parent shared libraries relocatable and hide linked archive internals.
function(tvm_configure_target_library target)
  if(APPLE)
    set_target_properties(${target} PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
  elseif(UNIX)
    set_target_properties(${target} PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
  endif()
  if(TVM_HIDE_PRIVATE_SYMBOLS
     AND CMAKE_SYSTEM_NAME MATCHES "Linux|Android|FreeBSD|NetBSD|OpenBSD"
     AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(${target} PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
endfunction()
