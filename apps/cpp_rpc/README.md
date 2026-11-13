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

# TVM RPC Server
This folder contains a simple recipe to make RPC server in c++.

## Usage (Non-Windows)

Configure from the TVM repository root with `cmake -S . -B build -DTVM_USE_CPP_RPC=ON`.
For a runtime-only build without LLVM or CUDA, also pass
`-DTVM_TIRX_USE_LLVM=OFF -DTVM_TIRX_USE_CUDA=OFF`.
For Android cross compilation, pass the toolchain and target options explicitly:

```bash
cmake -S . -B build -DTVM_USE_CPP_RPC=ON \
  -DTVM_TIRX_USE_LLVM=OFF -DTVM_TIRX_USE_CUDA=OFF \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28
```

For embedded Linux, configure with the cross compiler:

```bash
cmake -S . -B build -DTVM_USE_CPP_RPC=ON \
  -DTVM_TIRX_USE_LLVM=OFF -DTVM_TIRX_USE_CUDA=OFF \
  -DOS=Linux -DCMAKE_CXX_COMPILER=/path/to/cross/compiler/executable
```

For OpenCL support, add `-DTVM_TIRX_USE_OPENCL=ON`. The
[OpenCL wrapper](../../3rdparty/tvm-tirx/src/runtime/opencl/opencl_wrapper) dynamically loads
the device's OpenCL library when a system OpenCL SDK is unavailable. The device
must provide an OpenCL library at runtime. To link against a custom OpenCL SDK,
pass `-DTVM_TIRX_USE_OPENCL=/path/to/opencl-sdk`; that SDK should contain the
`include/CL` headers and `lib/` or `lib64/libOpenCL.so`.

Build the runtime and RPC server:

```bash
cmake --build build --target tvm_runtime tvm_rpc --parallel
```

Use `./build/tvm_rpc server` to start the RPC server.

## Usage (Windows)

- Configure the TVM build with `cmake -S . -B build -DTVM_USE_CPP_RPC=ON`.
- Install [LLVM pre-build binaries](https://releases.llvm.org/download.html), making sure to select the option to add it to the PATH.
- Verify Python 3.6 or newer is installed and in the PATH.
- Use `<tvm_output_dir>\tvm_rpc.exe` to start the RPC server.

## How it works
- The tvm runtime dll is linked along with this executable and when the RPC server starts it will load the tvm runtime library.

```
Command line usage
 server       - Start the server
--host        - The listen address of the server, Default=0.0.0.0 (any)
--port        - The port of the RPC server, Default=9090
--port-end    - The end search port of the RPC server, Default=9099
--tracker     - The RPC tracker address in host:port format e.g. 10.1.1.2:9190 Default=""
--key         - The key used to identify the device type in tracker. Default=""
--custom-addr - Custom IP Address to Report to RPC Tracker. Default=""
--silent      - Whether to run in silent mode. Default=False
  Example
  ./tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9099 --tracker=127.0.0.1:9190 --key=rasp
```

## Note
Currently support is only there for Linux / Android / Windows environment and proxy mode isn't supported currently.
