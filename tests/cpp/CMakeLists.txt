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

# Enable ctest if gtest is available
if(TVM_USE_GTEST AND NOT BUILD_TESTING STREQUAL "OFF")
  # Check env var for backward compatibility. A better way to specify package
  # locations is to use CMAKE_PREFIX_PATH or other standard CMake mechanism
  # (see CMake documentation for `find_package`).
  set(GTEST_ROOT "$ENV{GTEST_LIB}")
  if("${TVM_USE_GTEST}" STREQUAL "AUTO")
    # If TVM_USE_GTEST is AUTO, treat GTest as optional: enable if found.
    find_package(GTest)
  elseif("${TVM_USE_GTEST}" MATCHES ${IS_TRUE_PATTERN})
    # TVM_USE_GTEST is set to ON, TRUE, etc. Treat GTest as a required package.
    find_package(GTest REQUIRED)
  endif()
  if(GTEST_FOUND)
    if(NOT TARGET GTest::gmock)
      # GMock is formally supported in CMake 3.20; for now, expect libgmock.a in the same directory,
      # and require that folks compiling against GTest::gmock also link against GTest::GTest
      # (for the includes dir).
      add_library(GTest::gmock STATIC IMPORTED GLOBAL)
      get_target_property(GTEST_LIB_PATH GTest::GTest IMPORTED_LOCATION)
      if("${GTEST_LIB_PATH}" STREQUAL "GTEST_LIB_PATH-NOTFOUND")
        # CMake >= 3.20 makes GTest::GTest into a compatibility target. The real import location is in
        # GTest::gtest.
        get_target_property(GTEST_LIB_PATH GTest::gtest IMPORTED_LOCATION)
        if("${GTEST_LIB_PATH}" STREQUAL "GTEST_LIB_PATH-NOTFOUND")
          message(FATAL_ERROR "Neither GTest::GTest nor GTest::gtest targets defined IMPORTED_LOCATION")
        endif()
      endif()
      get_filename_component(GTEST_LIB_DIR "${GTEST_LIB_PATH}" DIRECTORY)
      set_target_properties(GTest::gmock PROPERTIES
          IMPORTED_LOCATION "${GTEST_LIB_DIR}/libgmock.a")
    endif()

    enable_testing()
    include(GoogleTest)
    set(GTEST_FOUND TRUE PARENT_SCOPE)
  endif()
endif()

# Create the `cpptest` target when GTest is available.
if(GTEST_FOUND)
  file(GLOB_RECURSE TEST_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
  add_executable(cpptest ${TEST_SRCS})
  tvm_configure_target(cpptest)
  # Preserve the executable location used before this became a subdirectory.
  set_target_properties(cpptest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
  # include runtime files for unit testing
  target_link_libraries(cpptest PRIVATE tvm_compiler GTest::GTest GTest::Main GTest::gmock pthread dl)
  if(DEFINED LLVM_LIBS)
    # The TVM library is linked with LLVM libraries. If the LLVM libraries are
    # static and the symbols are not hidden, then don't link them again into
    # cpptest since cpptest is itself linked against the TVM library. If static
    # LLVM libraries are linked in twice, it can cause issues with global
    # variable initialization (cl::opt).
    # If the LLVM libraries are dynamic, we have to link them again, since the
    # TVM library will not contain any LLVM definitions.
    unset(LLVM_SO)
    foreach(L IN LISTS LLVM_LIBS)
      if(L MATCHES "libLLVM.*\.so")
        set(LLVM_SO TRUE)
        break()
      endif()
    endforeach()
    if(DEFINED LLVM_SO OR TVM_HIDE_PRIVATE_SYMBOLS)
      target_link_libraries(cpptest PRIVATE ${LLVM_LIBS})
    endif()
  endif()
  set_target_properties(cpptest PROPERTIES EXCLUDE_FROM_ALL 1)
  set_target_properties(cpptest PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  # cpptest is built into ${CMAKE_BINARY_DIR} but the TVM shared libs it
  # links against (libtvm_compiler.so / libtvm_runtime.so) now live in
  # ${CMAKE_BINARY_DIR}/lib. CMake's default behaviour bakes the
  # absolute path of build/lib/ into the binary's RUNPATH, which works in tree
  # but is not relocatable. Set $ORIGIN/lib (or @loader_path/lib on macOS)
  # explicitly so the test binary finds its TVM libs relative to its own
  # location, mirroring the install-time RPATH setup.
  if(APPLE)
    set_target_properties(cpptest PROPERTIES
      BUILD_RPATH "@loader_path/lib"
      INSTALL_RPATH "@loader_path/lib"
    )
  elseif(UNIX)
    set_target_properties(cpptest PROPERTIES
      BUILD_RPATH "\$ORIGIN/lib"
      INSTALL_RPATH "\$ORIGIN/lib"
    )
  endif()
  target_compile_definitions(cpptest PRIVATE "NDEBUG")

  # For some reason, compile definitions are not propagated correctly, so we manually add them here
  target_compile_definitions(cpptest PUBLIC $<TARGET_PROPERTY:tvm_compiler,INTERFACE_COMPILE_DEFINITIONS>)
  gtest_discover_tests(cpptest WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()
