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

# iOS TVM RPC

This folder contains iOS RPC app that allows us to launch an rpc server on a iOS
device. You will need XCode and an iOS device to use this.

## Table of Contents
* [Building](#building)
    * [Building TVM runtime and custom DSO loader plugin](#building-tvm-runtime-and-custom-dso-loader-plugin)
    * [Building iOS TVM RPC application](#building-ios-tvm-rpc-application)
* [Workflow](#workflow)
    * [Standalone RPC](#standalone-rpc)
    * [iOS RPC App with proxy](#ios-rpc-app-with-proxy)
    * [iOS RPC App with tracker](#ios-rpc-app-with-tracker)
* [Communication without Wi-Fi and speed up in case of slow Wi-Fi](#communication-without-wi-fi-and-speed-up-in-case-of-slow-wi-fi)

## Building
### Building TVM runtime and custom DSO loader plugin
While iOS platform itself doesn't allow us to run an unsigned binary, there is a
partial ability to run JIT code on real iOS devices. While application is
running under debug session, system allows allocating memory with write and
execute permissions (a debugger requirement). So we can use this feature to
implement the `tvm.rpc.server.load_module` PackedFunc, used to load code over
RPC. For this purpose we use custom version of `dlopen` function which doesn't
check signature and permissions for module loading.  This custom `dlopen`
mechanic is integrated into TVM RPC as plugin and registered to execution only
inside iOS RPC application.

The custom implementation of `dlopen` and other functions from `dlfcn.h` header are placed in separate repository,
and will be downloaded automatically during CMake build for iOS.

Also, it is necessary to build `libtvm_runtime.dylib` for our iOS device. The
iOS TVM RPC application will be linked with this library.

Run the build using the following commands:
```shell
export DEVELOPER_DIR=/Applications/Xcode.app  # iOS SDK is part of Xcode bundle. Have to set it as default Dev Env
cmake ..
  -DCMAKE_BUILD_TYPE=Debug
  -DCMAKE_SYSTEM_NAME=iOS
  -DCMAKE_SYSTEM_VERSION=14.0
  -DCMAKE_OSX_SYSROOT=iphoneos
  -DCMAKE_OSX_ARCHITECTURES=arm64
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
  -DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=ON
  -DTVM_USE_IOS_RPC=ON  # to enable build iOS RPC application from TVM project tree
  -DTVM_TIRX_USE_METAL=ON    # to enable Metal runtime

cmake --build . --target custom_dso_loader tvm_runtime
```

### Building iOS TVM RPC application
Before start, please run [init_proj.py](./init_proj.py) to update XCode developer metadata:
```shell
python3 init_proj.py --team_id XXXXXXXXXX --tvm_build_dir "/path/to/tvm/ios/build/folder"
```
You can get value of your `team_id` in the following ways:
- **You have registered Apple Developer Profile**. In this case you developer
  Team ID available at https://developer.apple.com/account/#/membership
- You are using your local developer profile. In this case, leave `XXXXXXXXXX`
  in the command instead of substituting a Team ID. Then open `tvmrpc.xcodeproj`
  by using XCode, click on the project name (`tvmrpc`) on the left panel. Then
  select target `tvmrpc`. At the bottom of this panel go to `Signing &
  Capabilities` tab and in the field `Team` select your local developer profile
  (`Your Name (Personal Team)`).

  On the first run of the application you may see message `Could not launch
  "tvmrpc"` in the XCode and message `Untrusted Developer` on your device. In
  this case it will be necessary to check the certificate. Open
  `Settings -> General -> Device Management -> Apple Development: <your_email>
  -> Trust "Apple Development: <your_email>"` and click `Trust`. After than you
  should rerun your application in the XCode.

After this step, open `tvmrpc.xcodeproj` by using XCode, build the App and
install the App on the phone.

## Workflow
Due to security restriction of iOS10. We cannot upload dynamic libraries to the
App and load it from sandbox.  Instead, we need to build a list of libraries,
pack them into the app bundle, launch the RPC server and connect to test the
bundled libraries.  For more on the approach we use to work around this
limitation, please take a look into section
[Building TVM runtime and custom DSO loader integration](#building-tvm-runtime-and-custom-DSO-loader-plugin).

The test script [tests/ios_rpc_test.py](tests/ios_rpc_test.py) and
[tests/ios_rpc_mobilenet.py](tests/ios_rpc_mobilenet.py) are good templates for
demonstrating the workflow.

We have three different modes for iOS RPC server:
- [Standalone RPC](#standalone-rpc): In this mode RPC server open port on the device and listening. Then
  client connects to the server directly without any mediators.
- [iOS RPC application with Proxy](#ios-rpc-app-with-proxy): RPC server and RPC client communicates through
  `rpc_proxy`. The RPC server on iOS device notify `rpc_proxy` which was run on
  host machine about itself and wait for incoming connections. Communications
  between client and server works through `rpc_proxy`.
- [iOS RPC application with Tracker](#ios-rpc-app-with-tracker): RPC server registered in the `rpc_tracker`
  and client connects to the RPC server through `rpc_tracker`.

### Standalone RPC
Start RPC server on your iOS device:
- Push on the `Connect` button.

After that you supposed to see something like this in the app on the device:
```
IP: <device_ip>
Port: <rpc_server_port>
```

Printed `IP` is the IP address of your device and `PORT` is the number of port
which was open for RPC connection. Next you should use them for connect your RPC
client to the server.

Let's check that direct RPC connection works and we can upload a library with
model and execute it on the device. For this purpose we will use
[ios_rpc_test.py](tests/ios_rpc_test.py). Run it:
```shell
python3 tests/ios_rpc_test.py --host <device_ip> --port <rpc_server_port> --mode "standalone"
```
This will compile TVM IR to shared libraries (CPU and Metal) and run vector
addition on your iOS device. You are supposed to see something like this:
```
Metal: 0.000338692 secs/op
CPU: 0.000219308 secs/op
```

### iOS RPC App with proxy
Start the RPC proxy by running in a terminal:
```shell
python3 -m tvm.exec.rpc_proxy --host 0.0.0.0 --port 9090
```

On success, you should see something like this:
```
INFO:root:RPCProxy: client port bind to 0.0.0.0:9090
INFO:root:RPCProxy: Websock port bind to 8888
```
Connect your iOS device to the RPC proxy via the iOS TVM RPC application. Set
the `Address` and `Port` fields to the address and port of the RPC tracker
respectively. Select mode `Proxy` and push `Connect` button. In success the
text on the button will be changed to `Disconnect` and `Disconnected` in the top
of the screen will be changed to `Connected`.
On RPC proxy side you can see the next message in a log:
```
INFO:root:Handler ready TCPSocketProxy:<iPhone IP address>:server:iphone
```
Then we can check that RPC connection works and we can upload a library with
model and execute it on the target device. For this purpose we will use
[ios_rpc_test.py](tests/ios_rpc_test.py). Run it:
```shell
python3 tests/ios_rpc_test.py --host <host_ip_address> --port 9090 --mode "proxy"
```
The output should be the same as it was in previous section.

### iOS RPC App with tracker
First start an RPC tracker using
```shell
python3 -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190
```
On success, you should see something like this:
```
INFO:RPCTracker:bind to 0.0.0.0:9190
```
Connect your iOS device to the RPC tracker via the iOS TVM RPC applcation. Set
the `Address` and `Port` fields to the address and port of the RPC tracker
respectively. Select mode `Tracker` and push `Connect` button. In success the
text on the button will be changed to `Disconnect` and `Disconnected` in the top
of the screen will be changed to `Connected`. On the host side you can check the
connect by the following command:
```shell
python3 -m tvm.exec.query_rpc_tracker --port 9190
```
You are supposed to see something like this:
```
Tracker address 127.0.0.1:9190

Server List
----------------------------
server-address  key
----------------------------
192.168.1.57:9190       server:iphone
----------------------------

Queue Status
------------------------------
key      total  free  pending
------------------------------
iphone   1      1     0
------------------------------
```

Then we can check that RPC connection works and we can upload a library with
model and execute it on the target device. For this purpose we will use
[ios_rpc_test.py](tests/ios_rpc_test.py). Run it:
```shell
python3 tests/ios_rpc_test.py --host <host_ip_address> --port 9190 --mode "tracker"
```
The output will be the same as in section
[Standalone RPC](#standalone-rpc).

## Communication without Wi-Fi and speed up in case of slow Wi-Fi
Connection to the RPC server through `usbmux` can be used then you have slow,
unstable or don't have any Wi-Fi connection. `usbmux` is used for binding local
TCP port to port on the device and transfer packages between these ports by USB
cable.

First of all you should install `usbmux` to your system. You can do it with
brew:
```shell
brew install usbmuxd
```
After that you can use `iproxy` program for binding ports. You can use it for
all described workflows. Let's take a look how it works for
[Standalone RPC](#standalone-rpc).

First, start RPC server on your iOS device. You may see something like this in
the app on the device:
```
IP: unknown
Port: <rpc_server_port>
```
**Note.** Here `IP: unknown` because there was no Internet connection on the iOS
device.
Printed `Port` is the port of the RPC server on your iOS device. We will use it
in binding ports. Run `iproxy`, specify local port which should be used for
communication with device and the printed port on the device:
```shell
iproxy <local_port>:<rpc_server_port>
```
After this command you should see something like this:
```
Creating listening port <local_port> for device port <rpc_server_port>
waiting for connection
```
Now we can check that RPC connection through `usbmux` works and we can upload a
library with model and execute it on the device. For this purpose we will use
[ios_rpc_test.py](tests/ios_rpc_test.py). Run it:
```shell
python3 tests/ios_rpc_test.py --host 0.0.0.0 --port <local_port> --mode standalone
```
The output should be the same as in all previous runs.
