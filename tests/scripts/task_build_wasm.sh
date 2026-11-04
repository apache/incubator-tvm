#!/usr/bin/env bash
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

set -euxo pipefail

export SKBUILD_CMAKE_BUILD_TYPE=RelWithDebInfo
export SKBUILD_CMAKE_SOURCE_DIR="$PWD"
export SKBUILD_BUILD_DIR="$PWD/build"
export SKBUILD_BUILD_TARGETS="all;cpptest"
export CMAKE_ARGS="
    -DTVM_VERSION=$(python3 -m setuptools_scm)
    -DTVM_BUILD_PYTHON_MODULE=ON
    -DTVM_TIRX_BUILD_PYTHON_MODULE=ON
    -DTVM_FFI_BUILD_PYTHON_MODULE=ON
    -DTVM_FFI_BUILD_TESTS=OFF
    -DTVM_FFI_ATTACH_DEBUG_SYMBOLS=ON
    -DTVM_USE_SORT=ON
    -DTVM_TIRX_USE_LLVM=ON
    -DCMAKE_CXX_FLAGS=-Werror
    -DTVM_HIDE_PRIVATE_SYMBOLS=ON
    -DTVM_USE_CCACHE=OFF
    -DTVM_USE_INSTALLED_TIRX=OFF
    -DTVM_USE_GTEST=ON
    -DBUILD_TESTING=ON
    -DTVM_TIRX_BUILD_TESTS=ON
"
uv build --wheel --no-build-isolation --offline --python python3 --out-dir dist ./3rdparty/tvm-tirx/3rdparty/tvm-ffi
uv build --wheel --no-build-isolation --offline --python python3 --out-dir dist ./3rdparty/tvm-tirx
uv build --wheel --no-build-isolation --offline --python python3

./build/tirx_cpptest
./build/cpptest
