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

unset PYTHONPATH TVM_LIBRARY_PATH TVM_FFI_LIBRARY_PATH TVM_TIRX_LIBRARY_PATH
export PYTHONNOUSERSITE=1 TVM_HOME="$PWD"
uv pip install --python python3 --target .local/python --no-deps apache-tvm-ffi-orcjit==0.1.2
uv pip install --python python3 --target .local/python --no-deps --offline --reinstall \
    dist/apache_tvm-*.whl dist/apache_tvm_ffi-*.whl dist/apache_tvm_tirx-*.whl
export PYTHONPATH="$PWD/.local/python"
