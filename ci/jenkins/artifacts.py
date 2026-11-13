#!/usr/bin/env python3
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
"""Named CI artifacts. Uploads expand globs; downloads copy the saved paths.

Examples::

    python3 ci/jenkins/artifacts.py tvm_lib cpptest
    ./ci/jenkins/scripts/s3.py --action upload --bucket BUCKET --prefix PREFIX \
        --artifact tvm_lib --artifact cpptest
"""

import sys
from glob import glob

ARTIFACTS = {
    # Executables and build files needed to run c++ tests
    "cpptest": [
        "build/cpptest",
        "build/tirx_cpptest",
        "build/build.ninja",
        "build/CMakeFiles/rules.ninja",
    ],
    # runtime files
    "tvm_runtime": ["build/lib/libtvm_runtime.so", "build/CMakeCache.txt"],
    # compiler files
    "tvm_lib": [
        "build/lib/libtvm_compiler.so",
        "build/lib/libtvm_tirx_compiler.so",
        "build/lib/libtvm_runtime.so",
        "python/tvm_ffi",
        *glob("python/apache_tvm_ffi-*.dist-info"),
        "build/lib/libtvm_runtime_cuda.so",
        "build/lib/libtvm_runtime_vulkan.so",
        "build/lib/libtvm_runtime_opencl.so",
        "build/lib/libtvm_runtime_rocm.so",
        "build/lib/libtvm_runtime_extra.so",
        "build/libtvm_allvisible.so",
        "build/CMakeCache.txt",
    ],
    # gpu related compiler files
    "tvm_lib_gpu_extra": [
        "build/3rdparty/libflash_attn/src/libflash_attn.so",
        "build/3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/libfpA_intB_gemm.so",
    ],
}


if __name__ == "__main__":
    paths = []
    for name in sys.argv[1:]:
        if name not in ARTIFACTS:
            print(f"unknown bundle: {name}", file=sys.stderr)
            sys.exit(1)
        paths.extend(ARTIFACTS[name])
    for p in paths:
        print(p)
