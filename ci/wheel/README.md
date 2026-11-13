<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVM wheel packaging

The wheels are built by a standard `cibuildwheel` flow, configured in
`.github/workflows/publish_wheel.yml` and `3rdparty/tvm-tirx/pyproject.toml` (`[tool.cibuildwheel]`
and `[tool.scikit-build]`). This directory holds the few helper scripts that flow
invokes:

- `manylinux_build_libtvm_runtime_cuda.sh` — run by the `build_cuda_runtime` CI
  stage; builds the `libtvm_runtime_cuda.so` sidecar inside the prebuilt
  `quay.io/manylinux_cuda` image (CUDA toolkit preinstalled).
- `windows_build_libtvm_runtime_cuda.bat` — the Windows equivalent (run with
  `shell: cmd`); installs the CUDA toolkit via conda and builds
  `tvm_runtime_cuda.dll`.
- `build-environment.yaml` — conda environment for building the wheel.
