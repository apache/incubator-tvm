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

# Forget compiler wrappers cached by earlier dependency builds.
rm -f 3rdparty/tvm-tirx/3rdparty/tvm-ffi/build/CMakeCache.txt

rm -rf dist
uv pip install --python python3 --target .local/python \
    "scikit-build-core>=0.11,!=1.1.0" "setuptools-scm>=8" "cython>=3.2.8" "mlc-z3-static>=4.16.0" \
    "mlc-llvm-static[all]==23.1.1.post0"
export PYTHONPATH="$PWD/.local/python"
