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
# ruff: noqa: E501

import ctypes
import ctypes.util

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

# fmt: off


@I.ir_module
class Module:
    @R.function(pure=False)
    def main(x: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        cls = Module
        R.func_attr({"global_symbol": "main"})
        gv: R.Tuple(R.Any, R.Any) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.get_cached_alloc", (cls.cuda_graph_alloc, R.prim_value(0)), ty_args=(R.Tuple(R.Any, R.Any),))
        storage: R.Any = gv[0]
        alloc = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape((16, 16)), R.dtype("float32"))
        _: R.Tuple = R.call_tir_packed(cls.add, (x, alloc))
        storage1: R.Any = gv[1]
        gv1: R.Tuple(R.Tensor(dtype="float32"), R.Any, R.Any) = (alloc, storage1, storage)
        gv2: R.Tuple(R.Tensor((16, 16), dtype="float32")) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (cls.cuda_graph_capture, gv1, R.prim_value(0)), ty_args=(R.Tuple(R.Tensor((16, 16), dtype="float32")),))
        storage2: R.Any = R.vm.alloc_storage(R.shape((1024,)), R.prim_value(0), R.dtype("uint8"))
        alloc3 = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape((16, 16)), R.dtype("float32"))
        lv4: R.Tensor((16, 16), dtype="float32") = gv2[0]
        _3: R.Tuple = R.call_tir_packed(cls.add, (lv4, alloc3))
        lv5: R.Tensor(dtype="float32") = alloc3
        return lv5

    @Ts.function
    def add(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")):
        T.func_attr({"global_symbol": "add"})
        with Ts.sblock("root"):
            for i in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                for j in range(16):
                    with Ts.sblock("update"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] + T.float32(1)

    @R.function
    def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any):
        R.func_attr({"global_symbol": "cuda_graph_alloc"})
        storage: R.Any = R.vm.alloc_storage(R.shape((1024,)), R.prim_value(0), R.dtype("uint8"))
        storage1: R.Any = R.vm.alloc_storage(R.shape((1024,)), R.prim_value(0), R.dtype("uint8"))
        gv: R.Tuple(R.Any, R.Any) = (storage, storage1)
        return gv

    @R.function(pure=False)
    def cuda_graph_capture(alloc: R.Tensor((16, 16), dtype="float32"), storage1: R.Any, storage: R.Any) -> R.Tuple(R.Tensor((16, 16), dtype="float32")):
        cls = Module
        R.func_attr({"global_symbol": "cuda_graph_capture"})
        lv0: R.Tensor((16, 16), dtype="float32") = alloc
        alloc1 = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape((16, 16)), R.dtype("float32"))
        _1: R.Tuple = R.call_tir_packed(cls.add, (lv0, alloc1))
        lv1: R.Tensor(dtype="float32") = alloc1
        lv2: R.Tuple(R.Tensor(dtype="float32")) = (lv1,)
        lv3: R.Tensor(dtype="float32") = lv2[0]
        alloc2 = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape((16, 16)), R.dtype("float32"))
        _2: R.Tuple = R.call_tir_packed(cls.add, (lv3, alloc2))
        lv4: R.Tensor(dtype="float32") = alloc2
        gv: R.Tuple(R.Tensor(dtype="float32")) = (lv4,)
        return gv


# fmt: on


def codegen(mod, target):
    builder = relax.ExecBuilder()
    leftover_mod = relax.vm_build._vmcodegen(builder, mod)
    tir_mod = relax.vm_build._filter_tir(leftover_mod)
    return relax.vm_build._vmlink(builder, target, tir_mod)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_vm_run():
    mod = Module
    target = tvm_tirx.target.Target("cuda", host="llvm")
    ex = codegen(mod, target)
    x_np = np.random.uniform(size=(16, 16)).astype("float32")
    y_np = x_np + 1.0 + 1.0 + 1.0 + 1.0

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        vm = relax.VirtualMachine(ex, dev)
        x = tvm_tirx.runtime.tensor(x_np, dev)
        y = vm["main"](x)
        tvm_tirx.testing.assert_allclose(y.numpy(), y_np, rtol=1e-5, atol=1e-5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cudagraph(), reason="need cudagraph")
def test_capture_error_is_recoverable():
    """Function calls while capturing cudagraph may throw exceptions

    Calls to PackedFuncs may occur within a captured cudaGraph.  If a
    call to that PackedFunc raises an exception while capturing the
    cudaGraph, throwing exception should cleanly unwind the stack, and
    the exception may be caught in the calling scope.

    This is a regression test.  In previous implementations, an
    exception thrown while capturing a cudaGraph would skip the call
    to `cudaStreamEndCapture`, causing additional exceptions to be
    thrown while freeing memory in TVM destructors.  Since C++ does
    not support stack unwinding from multiple simultaneous exceptions,
    this would result in immediate `std::terminate`, making it
    difficult to debug the original error.

    """

    target = tvm_tirx.target.Target("cuda")

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.add(A, A)
            C = R.call_pure_packed(
                "test_vm_cuda_graph.invalid_impl_for_cudagraph",
                B,
                ty_args=R.Tensor([16], "float16"),
            )
            D = R.add(C, C)
            return D

    with tvm_tirx.ir.transform.PassContext(config={"relax.backend.use_cuda_graph": True}):
        Module = tvm_tirx.ir.transform.Sequential(
            [
                tvm.relax.transform.LegalizeOps(target=target),
                tvm.s_tir.transform.DefaultGPUSchedule(target=target),
                tvm.relax.transform.RemovePurityChecking(),
                tvm.relax.transform.CallTIRRewrite(),
                tvm.relax.transform.StaticPlanBlockMemory(),
                tvm.relax.transform.RewriteCUDAGraph(),
            ]
        )(Module)

    assert "cuda_graph_alloc" in Module, (
        "Validity of unit test requires the call to `invalid_impl_for_cudagraph` "
        "to have been captured by RewriteCUDAGraph."
    )

    built = tvm.compile(Module, target=target)

    def run_and_check():
        dev = tvm_tirx.cuda()
        cudart_path = ctypes.util.find_library("cudart")
        assert cudart_path is not None, "Unable to locate the CUDA runtime library"
        cudart = ctypes.CDLL(cudart_path)
        cudart.cudaGetLastError.argtypes = []
        cudart.cudaGetLastError.restype = ctypes.c_int
        cudart.cudaGetLastError()

        @tvm_ffi.register_global_func(
            "test_vm_cuda_graph.invalid_impl_for_cudagraph", override=True
        )
        def invalid_impl_for_cudagraph(arg_tensor):
            # Memory allocation/deallocation may not be performed while
            # capturing a cudaGraph.  This passes the warm-up run
            # performed by "vm.builtin.cuda_graph.run_or_capture", but
            # throws an exception when the cudaGraph is being captured.
            _dummy_workspace = tvm_tirx.runtime.empty([16], "float16", dev)
            return arg_tensor

        vm = tvm.relax.VirtualMachine(built, dev)
        arg = tvm_tirx.runtime.tensor(np.arange(16).astype("float16"), dev)

        with pytest.raises(RuntimeError):
            vm["main"](arg)

        # cudaGetLastError is host-thread-local, so query it in the same
        # callback that triggered the invalid capture.
        cuda_error = cudart.cudaGetLastError()
        assert cuda_error == 0, (
            f"CUDA error state was not cleared after failed graph capture: {cuda_error}"
        )

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
