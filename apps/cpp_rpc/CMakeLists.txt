cmake_policy(SET CMP0069 NEW) # suppress CMake warning about IPO

set(TVM_RPC_SOURCES
  main.cc
  rpc_env.cc
  rpc_server.cc
)

set(TVM_RPC_LINKER_LIBS "")

if(WIN32)
  list(APPEND TVM_RPC_SOURCES win32_process.cc)
endif()

# Set output to same directory as the other TVM libs
add_executable(tvm_rpc ${TVM_RPC_SOURCES})
tvm_configure_target(tvm_rpc)
set_target_properties(tvm_rpc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET tvm_rpc PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(WIN32)
  target_compile_definitions(tvm_rpc PUBLIC -DNOMINMAX)
endif()

if (OS)
   if (OS STREQUAL "Linux")
      set_property(TARGET tvm_rpc PROPERTY LINK_FLAGS -lpthread)
   endif()
endif()

if(TARGET tvm_runtime_opencl)
   if (ANDROID_ABI)
     if(DEFINED ENV{ANDROID_NDK_MAJOR})
       if($ENV{ANDROID_NDK_MAJOR} VERSION_LESS "23")
         set_property(TARGET tvm_rpc PROPERTY LINK_FLAGS -fuse-ld=gold)
       endif()
     endif()
   endif()
endif()

target_include_directories(
  tvm_rpc
  PUBLIC "../../include"
)

target_link_libraries(tvm_rpc PUBLIC tvm_ffi::shared)

if(NOT MSVC AND NOT APPLE)
  list(APPEND TVM_RPC_LINKER_LIBS tvm_runtime "-Wl,--no-as-needed" ${TVM_RUNTIME_BACKEND_LIBS} "-Wl,--as-needed")
else()
  list(APPEND TVM_RPC_LINKER_LIBS tvm_runtime ${TVM_RUNTIME_BACKEND_LIBS})
endif()

target_link_libraries(tvm_rpc PRIVATE ${TVM_RPC_LINKER_LIBS})
