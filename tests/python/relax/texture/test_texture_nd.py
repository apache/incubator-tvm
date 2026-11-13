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
# ruff: noqa: F401, F821, F841

import os
import tempfile

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno
from tvm.rpc import connect_tracker
from tvm.s_tir import script as Ts
from tvm.testing import env as tvm_env
from tvm_tirx import IRModule, tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support import ndk
from tvm_tirx.target import Target


def get_rpc():
    """
    Establish an RPC connection to the remote device.

    Returns
    -------
    tvm.rpc.RPCSession or None
        The RPC session object if RPC_TARGET is set; otherwise, None.
    """
    rpc_target = os.getenv("RPC_TARGET", None)
    if rpc_target:
        host = os.getenv("TVM_TRACKER_HOST", "localhost")
        port = int(os.getenv("TVM_TRACKER_PORT", 9090))
        device_key = os.getenv("RPC_DEVICE_KEY", "android")
        tracker = connect_tracker(host, port)
        return tracker.request(device_key, priority=1, session_timeout=1000)
    else:
        return None


def preprocess_pipeline(mod: IRModule, *, target=None) -> IRModule:
    desired_layouts = {"relax.nn.conv2d": ["NCHW16c", "OIHW16o", "NCHW16c"]}
    seq = tvm_tirx.ir.transform.Sequential(
        [
            tvm_tirx.tirx.transform.BindTarget(target),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.DecomposeOpsForInference(),
            tvm.relax.transform.FoldConstant(),
            tvm_tirx.tirx.transform.BindTarget(target),
            tvm.relax.transform.ConvertLayout(desired_layouts),
            tvm.relax.transform.Normalize(),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.LegalizeOps(),
            tvm.relax.transform.LegalizeOps(
                {"relax.nn.conv2d": legalize_adreno.conv2d_NCHWc_OIHWo}
            ),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.AnnotateTIROpPattern(),
            tvm.relax.transform.FuseOps(),
            tvm.relax.transform.FuseTIR(),
            tvm.relax.transform.DeadCodeElimination(),
            tvm.relax.transform.Normalize(),
        ]
    )
    mod = seq(mod)
    return mod


def postprocess_pipeline(mod: IRModule, *, target=None) -> IRModule:
    seq = tvm_tirx.ir.transform.Sequential(
        [
            tvm.relax.transform.ToNonDataflow(),
            tvm.relax.transform.RemovePurityChecking(),
            tvm.relax.transform.CallTIRRewrite(),
            tvm.relax.transform.Normalize(),
            tvm.relax.transform.StaticPlanBlockMemory(),
            tvm.relax.transform.LowerAllocTensor(),
            tvm.relax.transform.KillAfterLastUse(),
            tvm.relax.transform.LowerRuntimeBuiltin(),
            tvm.relax.transform.VMShapeLower(),
            tvm.relax.transform.AttachGlobalSymbol(),
        ]
    )
    mod = seq(mod)
    return mod


@pytest.mark.skipif(not tvm_env.build_flag_enabled("USE_RPC"), reason="need rpc")
@pytest.mark.gpu
@pytest.mark.skipif(not tvm_env.has_adreno_opencl(), reason="need adreno opencl")
@pytest.mark.parametrize("backend", ["opencl"])
@pytest.mark.parametrize("dtype", ["int8", "float16", "int16", "float32", "int32"])
@pytest.mark.parametrize("channel_size", [64, 128])
@pytest.mark.parametrize("read_width", [1, 2, 4, 8, 16])
def test_texture_copy(backend, dtype, channel_size, read_width):
    remote = get_rpc()
    M, N, K = (256, 1024, 128)
    lanes = channel_size // tvm_ffi.dtype(dtype).bits
    if read_width > lanes:
        return

    @I.ir_module
    class TextureCopy:
        @Ts.function
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
            T.func_attr({"global_symbol": "main"})
            for li, lj in T.grid(M, N):
                with Ts.sblock("Copy"):
                    i, j = Ts.axis.remap("SS", [li, lj])
                    B[i, j] = A[i, j]

    def schedule_texture_read(sch: s_tir.Schedule):
        B_blk = sch.get_sblock("Copy")
        Ai_block = sch.cache_read(B_blk, 0, "global.texture")
        sch.transform_layout(Ai_block, ("write", 0), lambda i, j: (i, j // lanes, j % lanes))

        def schedule_default(blk, lanes):
            i, j = sch.get_loops(blk)
            jo, jv = sch.split(j, [None, lanes])

            b = sch.fuse(i, jo)
            bx, tx = sch.split(b, [None, 256])
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

            sch.vectorize(jv)

        schedule_default(Ai_block, lanes)
        schedule_default(B_blk, read_width)

    mod = TextureCopy

    if remote is None:
        target = Target({"kind": backend, "device": "adreno"})
    else:
        target = Target(
            {"kind": backend, "device": "adreno"},
            {"kind": "llvm", "mtriple": "aarch64-linux-android"},
        )

    mod = preprocess_pipeline(mod, target=target)
    sch = tvm.s_tir.Schedule(mod)
    schedule_texture_read(sch)
    mod = postprocess_pipeline(sch.mod, target=target)

    ex = relax.build(mod, target)
    load_path = "vm_library.so"
    inputs = [np.random.randint(0, 128, (M, N)).astype(dtype), np.zeros((M, N), dtype)]

    def run_and_check(rexec, remote_session):
        if remote_session is not None:
            dev = remote_session.cl()
        elif "opencl" in backend:
            dev = tvm_tirx.opencl(0)
        elif "vulkan" in backend:
            dev = tvm_tirx.vulkan(0)
        else:
            raise RuntimeError("Unsupported backend")

        if "vdevice" in mod.global_infos:
            device_arr = [dev for _ in range(len(mod.global_infos["vdevice"]))]
        else:
            device_arr = [dev]
        if remote_session is not None:
            vm = tvm.rpc.virtual_machine(rexec, device_arr)
            inps = [dev.tensor(inp) for inp in inputs]
        else:
            vm = relax.VirtualMachine(rexec, device_arr)
            inps = [tvm_tirx.runtime.tensor(inp, dev) for inp in inputs]
        vm["main"](*inps)
        np.testing.assert_equal(inps[-1].numpy(), inps[0].numpy())

    with tempfile.TemporaryDirectory() as temp_dir:
        if remote is not None:
            path = temp_dir + "/" + load_path
            ex.export_library(path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"])
            remote.upload(path)
            rexec = remote.load_module(load_path)
            try:
                run_and_check(rexec, remote)
            finally:
                remote.get_function("CloseRPCConnection")()
        else:
            tvm_tirx.testing.run_with_gpu_lock(run_and_check, ex, None)


if __name__ == "__main__":
    tvm_tirx.testing.main()
