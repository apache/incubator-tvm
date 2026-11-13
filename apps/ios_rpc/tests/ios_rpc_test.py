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
"""Testcode for iOS RPC.

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
And configure the proxy host field as commented.
"""

import argparse

import numpy as np
import tvm_ffi

import tvm
import tvm_tirx
from tvm import rpc, te
from tvm_tirx.support import utils, xcode

# Change target configuration, this is setting for iphone6s
arch = "arm64"
sdk = "iphoneos"
target = {"kind": "llvm", "mtriple": f"{arch}-apple-darwin"}

MODES = {"proxy": rpc.connect, "tracker": rpc.connect_tracker, "standalone": rpc.connect}


# override metal compiler to compile to iphone
@tvm_ffi.register_global_func("tvm_callback_metal_compile")
def compile_metal(src, target):
    return xcode.compile_metal(src, sdk=sdk)


def test_rpc_module(host, port, key, mode):
    # graph
    n = tvm_ffi.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    temp = utils.tempdir()
    mod = tvm_tirx.IRModule.from_expr(
        te.create_function([A, B]).with_attr("global_symbol", "myadd")
    )
    sch = tvm.s_tir.Schedule(mod)
    (i,) = sch.get_loops(block=sch.get_sblock("B"))
    i0, i1 = sch.split(i, [None, 32])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")

    # Build the dynamic lib.
    # If we don't want to do metal and only use cpu, just set target to be target
    f = tvm.compile(sch.mod, target=tvm_tirx.target.Target("metal", host=target))
    path_dso1 = temp.relpath("dev_lib.dylib")
    f.export_library(path_dso1, fcompile=xcode.create_dylib, arch=arch, sdk=sdk)

    # connect to the proxy
    if mode == "tracker":
        remote = MODES[mode](host, port).request(key)
    else:
        remote = MODES[mode](host, port, key=key)
    remote.upload(path_dso1)
    dev = remote.metal(0)
    f1 = remote.load_module("dev_lib.dylib")
    a_np = np.random.uniform(size=1024).astype(A.dtype)
    a = dev.tensor(a_np)
    b = dev.tensor(np.zeros(1024, dtype=A.dtype))
    time_f = tvm.runtime.time_evaluator(f1, f1.entry_name, dev, number=10)
    cost = time_f(a, b).mean
    print(f"Metal: {cost:g} secs/op")
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo app demonstrates how ios_rpc works.")
    parser.add_argument("--host", required=True, type=str, help="Address of rpc server")
    parser.add_argument("--port", type=int, default=9090, help="rpc port (default: 9090)")
    parser.add_argument("--key", type=str, default="iphone", help="device key (default: iphone)")
    parser.add_argument(
        "--mode",
        type=str,
        default="tracker",
        help="type of RPC connection (default: tracker), possible values: {}".format(
            ", ".join(MODES.keys())
        ),
    )

    args = parser.parse_args()
    assert args.mode in MODES.keys()
    test_rpc_module(args.host, args.port, args.key, args.mode)
