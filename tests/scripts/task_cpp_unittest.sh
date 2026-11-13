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

if [ $# -gt 0 ]; then
    BUILD_DIR="$1"
elif [ -n "${TVM_BUILD_PATH:-}" ]; then
    # TVM_BUILD_PATH may contain multiple space-separated paths.  If
    # so, use the first one.
    BUILD_DIR=$(IFS=" "; set -- $TVM_BUILD_PATH; echo $1)
else
    BUILD_DIR=build
fi

# to avoid CI thread throttling.
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1

# Test workers receive the exact FFI package in their own checkout. Native
# binaries must find that copy even when their build workspace had another path.
ffi_lib_dir="$(pwd)/python/tvm_ffi/lib"
export LD_LIBRARY_PATH="${ffi_lib_dir}:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${ffi_lib_dir}:${DYLD_LIBRARY_PATH:-}"

pushd "${BUILD_DIR}"
# run cpp test executable
./cpptest
./tirx_cpptest
popd
