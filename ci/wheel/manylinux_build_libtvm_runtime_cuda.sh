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
#
# Build libtvm_runtime_cuda.so inside a manylinux CUDA container, run by the
# build_cuda_runtime CI job. The official quay.io/manylinux_cuda images ship
# the CUDA toolkit preinstalled under /usr/local/cuda, but omit the libcuda
# driver stub, so we install just cuda-driver-devel from the image's CUDA repo.
# Builds the sidecar into build-wheel-cuda/lib/ for the wheel build to bundle.
#
# Usage: manylinux_build_libtvm_runtime_cuda.sh
set -euxo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${repo_root}/build-wheel-cuda"
parallel="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# The image ships the toolkit but not the libcuda driver stub; install it from
# the image's CUDA repo so the sidecar links libcuda.so.1 (sync version with tag).
dnf -y install cuda-driver-devel-13-1

# Build the CUDA runtime sidecar with CUDA on and LLVM off, so it does not need
# the LLVM prefix; the main CPU wheel links LLVM statically. The manylinux CUDA
# image already ships cmake and make, and the build uses the default Makefiles
# generator (no Ninja), so no build tools are installed here. Put the bundled
# CPython and CUDA toolchain on PATH for the CMake configure and nvcc.
export PATH="/opt/python/cp310-cp310/bin:/usr/local/cuda/bin:${PATH}"
nvcc --version

rm -rf "${build_dir}"
# CMAKE_CUDA_COMPILER only tells CMake which nvcc to use; it does not affect the
# resulting libtvm_runtime_cuda.so, which is built only from .cc host sources (no
# .cu device code, so nvcc is never invoked for it). CMAKE_CUDA_ARCHITECTURES is
# intentionally not set: it would be a no-op here for the same reason (verified --
# the .so is byte-identical across arch values and carries no device code), and
# modern CMake fills in a default so configure does not fail without it.
cmake -S "${repo_root}/3rdparty/tvm-tirx" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTVM_TIRX_BUILD_TESTS=OFF \
  -DTVM_TIRX_BUILD_PYTHON_MODULE=ON \
  -DTVM_TIRX_USE_CUDA=/usr/local/cuda \
  -DTVM_TIRX_USE_LLVM=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build "${build_dir}" --target tvm_runtime tvm_runtime_cuda --parallel "${parallel}"

cuda_lib="${build_dir}/lib/libtvm_runtime_cuda.so"
test -f "${cuda_lib}"
patchelf --set-rpath '$ORIGIN' "${cuda_lib}"
echo "CUDA runtime: ${cuda_lib}"
