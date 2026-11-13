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

# TVM4J - Java Frontend for TVM Runtime

This folder contains the Java interface for TVM runtime. It brings TVM runtime to Java virtual machine.

- It enables you to construct Tensor from Java native array and vice versa.
- You can register and convert Java native functions to TVM functions.
- It enables you to load shared libraries created by Python and C++.
- It provides a simple interface for RPC server and client.

## Installation

### Requirements

- JDK 1.6+. Oracle JDK and OpenJDK are well tested.
- Maven 3 for build.
- LLVM (TVM4J need LLVM support. Please refer to [build-the-shared-library](https://tvm.apache.org/docs/install/from_source.html#build-the-shared-library) for how to enable LLVM support.)

### Modules

TVM4J contains three modules:

- core
    * It contains all the Java interfaces.
- native
    * The JNI native library is compiled in this module using the installed TIRx and FFI packages.
- assembly
    * It assembles Java interfaces (core), JNI library (native) and TVM runtime library together. The simplest way to integrate tvm4j in your project is to rely on this module. It automatically extracts the native library to a tempfile and load it.

### Build

Build the FFI, TIRx, and TVM wheels for your platform first; the
[contributor guide](../docs/contribute/pull_request.rst) has the CI Docker build
commands. From the repository root, install the wheels from `dist/` and compile
tvm4j in the same shell:

```bash
source tests/scripts/test_task_install_dist_wheels.sh
bash tests/scripts/task_jvm_build.sh
```

The build discovers native headers and libraries through `tvm_tirx.config`
and `tvm_ffi.config`. The sourced helper sets `PYTHONPATH` to the installed
packages in `.local/python`.

To run the Java CI task in that shell:

```bash
bash tests/scripts/task_java_unittest.sh
```

After it is compiled and packaged, you can install tvm4j in your local maven repository,

```bash
bash tests/scripts/task_jvm_build.sh install
```

## Convert and Register Java Function as TVM Function

It is easy to define a Java function and call it from TVM. The following snippet demonstrate how to concatenate Java strings.

```java
Function func = Function.convertFunc(new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        StringBuilder res = new StringBuilder();
        for (TVMValue arg : args) {
          res.append(arg.asString());
        }
        return res.toString();
      }
    });
TVMValue res = func.pushArg("Hello").pushArg(" ").pushArg("World!").invoke();
assertEquals("Hello World!", res.asString());
res.release();
func.release();
```

It is your job to verify the types of callback arguments, as well as the type of returned result.

You can register the Java function by `Function.register` and use `Function.getFunction` to get the registered function later.

## Run the Generated Shared Library

The following code snippet demonstrate how to load generated shared library (add_cpu.so).

```java
import org.apache.tvm.Module;
import org.apache.tvm.Tensor;
import org.apache.tvm.Device;

import java.io.File;
import java.util.Arrays;

public class LoadAddFunc {
  public static void main(String[] args) {
    String loadingDir = args[0].cast<String>();
    Module fadd = Module.load(loadingDir + File.separator + "add_cpu.so");

    Device dev = Device.cpu();

    long[] shape = new long[]{2};
    Tensor arr = Tensor.empty(shape, dev);
    arr.copyFrom(new float[]{3f, 4f});
    Tensor res = Tensor.empty(shape, dev);

    fadd.entryFunc().pushArg(arr).pushArg(arr).pushArg(res).invoke();
    System.out.println(Arrays.toString(res.asFloatArray()));

    arr.release();
    res.release();
    fadd.release();
  }
}
```

## RPC Server

There are two ways to start an RPC server on JVM. A standalone server can be started by

```java
Server server = new Server(port);
server.start();
```

This will open a socket and wait for remote requests. You can use Java, Python, or any other frontend to make an RPC call. Here's an example for calling remote function `test.rpc.strcat` in Java.

```java
RPCSession client = Client.connect("127.0.0.1", port.value);
Function func = client.getFunction("test.rpc.strcat");
String result = func.call("abc", 11L).asString();
```

Another way is to start a proxy, make server and client communicate with each other through the proxy. The following snippet shows how to start a server which connects to a proxy.

```java
Server server = new Server(proxyHost, proxyPort, "key");
server.start();
```

You can also use `StandaloneServerProcessor` and `ConnectProxyServerProcessor` to build your own RPC server. Refer to [Android RPC Server](https://github.com/apache/tvm/blob/main/apps/android_rpc/app/src/main/java/org/apache/tvm/tvmrpc/RPCProcessor.java) for more details.
