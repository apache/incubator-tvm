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

from __future__ import annotations

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax, topi
from tvm.contrib.thrust import can_use_thrust
from tvm.relax import script as R
from tvm.relax.backend import DispatchSortScan
from tvm.s_tir import dlight
from tvm.testing import device_enabled
from tvm_tirx import tirx
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_dispatch_scanop():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "llvm")):
            with R.dataflow():
                lv0 = R.cumsum(x, axis=1, dtype="float64", exclusive=False)
                lv1 = R.cumprod(lv0, axis=1, dtype="float64", exclusive=False)
                gv = lv1
                R.output(gv)
            return gv

    mod = DispatchSortScan()(Before)

    vdevices = [R.vdevice("llvm", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()

    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.cumsum, x, axis=1, dtype="float64", exclusive=False)
            out = bb.emit_te(topi.cumprod, lv0, axis=1, dtype="float64", exclusive=False)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    assert_structural_equal(mod, expected_mod)


def test_dispatch_scanop_cuda():
    """R.cumsum and R.cumprod may be lowered with TOPI for GPU

    For the purpose of testing, this test case intentionally uses the
    `exclusive=True` argument to prevent the `R.cumsum` from being
    lowered to the packed func `"gpu_2d_continuous_cumsum"`.
    """

    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("cuda", 0)]})

        @R.function
        def main(x: R.Tensor((m, 3), "float32", "cuda")):
            with R.dataflow():
                lv0 = R.cumsum(x, axis=1, exclusive=True)
                lv1 = R.cumprod(lv0, axis=1)
                gv = lv1
                R.output(gv)
            return gv

    target = tvm_tirx.target.Target("cuda", host="llvm")

    vdevices = [R.vdevice("cuda", 0)]
    m = T.dynamic("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()
    with bb.function("main", (x,), {"global_symbol": "main"}):
        with bb.dataflow():
            lv = bb.emit_te(topi.gpu.cumsum, x, axis=1, exclusive=True, target=target)
            out = bb.emit_te(topi.gpu.cumprod, lv, axis=1, target=target)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan(target=target)(Before)
    expected_mod = dlight.ApplyDefaultSchedule(dlight.gpu.Fallback(), target=target)(expected_mod)

    assert_structural_equal(mod, expected_mod, map_free_vars=True)


def test_dispatch_sort():
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor((m, 3), "float32", "llvm")):
            with R.dataflow():
                lv = R.sort(x, axis=1, descending=False)
                gv = lv
                R.output(gv)
            return gv

    vdevices = [R.vdevice("llvm", 0)]
    m = T.dynamic("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()

    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            out = bb.emit_te(topi.sort, x, axis=1, is_ascend=True)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, expected_mod)


@pytest.mark.xfail(reason="skipping broken tests")
def test_dispatch_sort_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("cuda")]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "cuda")):
            with R.dataflow():
                lv = R.sort(x, axis=1, descending=False)
                gv = lv
                R.output(gv)
            return gv

        @R.function
        def foo2(y: R.Tensor((2, 3), "float32")):
            with R.dataflow():
                lv = R.sort(y, axis=0, descending=True)
                gv = lv
                R.output(gv)
            return gv

    target = tvm_tirx.target.Target({"kind": "cuda", "libs": ["thrust"]}, host="llvm")

    vdevices = [R.vdevice("cuda", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            out = bb.emit_te(topi.gpu.sort, x, axis=1, target=target)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    with bb.function("foo2", (y,), {"global_symbol": "foo2"}):
        with bb.dataflow():
            if can_use_thrust(target, "tvm.contrib.thrust.sort"):
                workspace = bb.emit(
                    relax.op.builtin.alloc_tensor(
                        relax.ShapeExpr([4194568]), "uint8", runtime_device_index=0
                    )
                )
                out = bb.emit_te(
                    topi.gpu.sort_thrust,
                    y,
                    axis=0,
                    is_ascend=False,
                    workspace=workspace,
                    target=target,
                )
            else:
                out = bb.emit_te(topi.gpu.sort, y, axis=0, is_ascend=False, target=target)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan(target=target)(Before)

    assert_structural_equal(mod, expected_mod)


def test_dispatch_argsort():
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor((m, 3), "float32", "llvm")):
            with R.dataflow():
                lv = R.argsort(x, axis=1, descending=False, dtype="int32")
                gv = lv
                R.output(gv)
            return gv

    vdevices = [R.vdevice("llvm", 0)]
    m = T.dynamic("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()

    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            out = bb.emit_te(topi.argsort, x, axis=1, is_ascend=True, dtype="int32")
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, expected_mod)


def test_dispatch_argsort_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("cuda")]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "cuda")):
            with R.dataflow():
                lv = R.argsort(x, axis=1, descending=False)
                gv = lv
                R.output(gv)
            return gv

        @R.function
        def foo2(y: R.Tensor((2, 3), "float32")):
            with R.dataflow():
                lv = R.argsort(y, axis=0, descending=True, dtype="int64")
                gv = lv
                R.output(gv)
            return gv

    target = tvm_tirx.target.Target({"kind": "cuda", "libs": ["thrust"]}, host="llvm")

    vdevices = [R.vdevice("cuda", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder(tvm_tirx.IRModule(global_infos={"vdevice": vdevices}))
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            out = bb.emit_te(
                topi.gpu.argsort, x, axis=1, is_ascend=True, dtype="int32", target=target
            )
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    with bb.function("foo2", (y,), {"global_symbol": "foo2"}):
        with bb.dataflow():
            if can_use_thrust(target, "tvm.contrib.thrust.sort"):
                workspace = bb.emit(
                    relax.op.builtin.alloc_tensor(
                        R.shape([8388872]), R.dtype("uint8"), R.prim_value(0), R.str("global")
                    )
                )
                out = bb.emit_te(
                    topi.gpu.argsort_thrust,
                    y,
                    axis=0,
                    is_ascend=False,
                    dtype="int64",
                    workspace=workspace,
                    target=target,
                )
            else:
                out = bb.emit_te(
                    topi.gpu.argsort, y, axis=0, is_ascend=False, dtype="int64", target=target
                )
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan(target=target)(Before)

    assert_structural_equal(mod, expected_mod)


def test_dispatch_topk():
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor((m, 3), "float32", "llvm")):
            with R.dataflow():
                lv = R.topk(x, k=2, axis=1, largest=True)
                gv = lv
                R.output(gv)
            return gv

    vdevices = [R.vdevice("llvm", 0)]
    m = T.dynamic("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()

    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            out = bb.emit_te(topi.topk, x, k=2, axis=1, is_ascend=False, dtype="int32")
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, expected_mod)


def test_dispatch_topk_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("cuda")]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "cuda")):
            with R.dataflow():
                lv = R.topk(x, k=2, axis=1, largest=True)
                gv = lv
                R.output(gv)
            return gv

    target = tvm_tirx.target.Target({"kind": "cuda", "libs": ["thrust"]}, host="llvm")

    vdevices = [R.vdevice("cuda", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            out = bb.emit_te(
                topi.gpu.topk, x, k=2, axis=1, is_ascend=False, dtype="int32", target=target
            )
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan(target=target)(Before)
    expected_mod = dlight.ApplyDefaultSchedule(dlight.gpu.Fallback(), target=target)(expected_mod)

    assert_structural_equal(mod, expected_mod)


def test_dispatch_topk_gpu():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("vulkan")]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "vulkan")):
            with R.dataflow():
                # Two same calls should have only one Function
                lv0 = R.topk(x, k=2, axis=1, largest=True)
                lv1 = R.topk(x, k=2, axis=1, largest=True)
                gv = (lv0, lv1)
                R.output(gv)
            return gv

    target = tvm_tirx.target.Target("vulkan", host="llvm")

    vdevices = [R.vdevice("vulkan", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            lv0 = bb.emit_te(
                topi.gpu.topk, x, k=2, axis=1, is_ascend=False, dtype="int32", target=target
            )
            lv1 = bb.emit_te(
                topi.gpu.topk, x, k=2, axis=1, is_ascend=False, dtype="int32", target=target
            )
            out = (lv0, lv1)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan(target=target)(Before)
    expected_mod = dlight.ApplyDefaultSchedule(dlight.gpu.Fallback(), target=target)(expected_mod)

    assert_structural_equal(mod, expected_mod)


@pytest.mark.gpu
@pytest.mark.parametrize("size", [300, 600])
def test_dispatch_sort_cuda_large_batch(size):
    target = "cuda"
    if not device_enabled(target):
        pytest.skip(f"{target} not enabled")

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((m, n), "float32")):
            with R.dataflow():
                gv = R.sort(x, axis=-1, descending=False)
                R.output(gv)
            return gv

    batch = 65600  # > CUDA's 65535 gridDim.y/z limit
    np_data = np.random.uniform(size=(batch, size)).astype("float32")
    np_sorted = np.sort(np_data, axis=-1)

    mod = DispatchSortScan(target=tvm_tirx.target.Target(target))(Module)
    ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm_tirx.device_from_target(target)
        vm = tvm.relax.VirtualMachine(ex, dev)
        tvm_data = tvm_tirx.runtime.tensor(np_data, dev)
        sorted_data = vm["main"](tvm_data)
        tvm_tirx.testing.assert_allclose(sorted_data.numpy(), np_sorted)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
def test_dispatch_topk_cuda_large_batch():
    target = "cuda"
    if not device_enabled(target):
        pytest.skip(f"{target} not enabled")

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((m, n), "float32")):
            with R.dataflow():
                gv = R.topk(x, k=1, axis=-1, ret_type="values", largest=True)
                R.output(gv)
            return gv

    batch, size = 65600, 300
    np_data = np.random.uniform(size=(batch, size)).astype("float32")
    np_values = np.sort(np_data, axis=-1)[:, -1:]

    mod = DispatchSortScan(target=tvm_tirx.target.Target(target))(Module)
    ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm_tirx.device_from_target(target)
        vm = tvm.relax.VirtualMachine(ex, dev)
        tvm_data = tvm_tirx.runtime.tensor(np_data, dev)
        values = vm["main"](tvm_data)
        tvm_tirx.testing.assert_allclose(values.numpy(), np_values)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.skipif(not tvm_tirx.testing.env.has_llvm(), reason="need llvm for the host module")
@pytest.mark.parametrize("op", ["sort", "argsort", "topk"])
@pytest.mark.parametrize("size", [300, None])
def test_dispatch_sort_webgpu_merge_grid(op, size):
    """Merge passes must not use blockIdx.z, which WebGPU reserves to extend blockIdx.x."""
    n = size if size is not None else tirx.Var("n", "int64")
    x = relax.Var("x", relax.TensorType((tirx.Var("m", "int64"), n), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        if op == "sort":
            out = relax.op.sort(x, axis=-1, descending=False)
        elif op == "argsort":
            out = relax.op.argsort(x, axis=-1, descending=True, dtype="int32")
        else:
            out = relax.op.topk(x, k=4, axis=-1, ret_type="values", largest=True)
        bb.emit_func_output(bb.emit(out))

    target = tvm_tirx.target.Target("webgpu", host="llvm")
    mod = DispatchSortScan(target=target)(bb.get())
    tvm.compile(mod, target)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("metal", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "supports_int64": True}, marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("index_bits", [None, 64])
def test_dispatch_cumsum_gpu(target, index_bits):
    """Test cumsum kernel dispatch and numerical correctness"""
    if not device_enabled(target):
        pytest.skip(f"{target} not enabled")

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((m, n), "int32")):
            with R.dataflow():
                gv = R.cumsum(x, axis=-1, exclusive=False)
                R.output(gv)
            return gv

    size = (8, 2000)
    np_data = np.random.randint(0, 10, size).astype("int32")
    np_cumsum = np.cumsum(np_data, axis=-1)
    mod = DispatchSortScan(index_bits=index_bits, target=tvm_tirx.target.Target(target))(Module)
    ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm_tirx.device_from_target(target)
        vm = tvm.relax.VirtualMachine(ex, dev)
        tvm_data = tvm_tirx.runtime.tensor(np_data, dev)
        cumsum = vm["main"](tvm_data)
        tvm_tirx.testing.assert_allclose(cumsum.numpy(), np_cumsum)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("target_kind", ["metal", "cuda"])
@pytest.mark.parametrize("index_bits", [None, 64])
def test_dispatch_cumsum_index_width(target_kind, index_bits):
    """Respect the caller's index budget without restricting Metal's default."""
    from tvm.relax.backend.gpu_generic import gpu_2d_continuous_cumsum

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((m, n), "float32")):
            gv = R.cumsum(x, axis=-1)
            return gv

    mod = DispatchSortScan(
        index_bits=index_bits, target=tvm_tirx.target.Target(target_kind, host="llvm")
    )(Module)

    expected = gpu_2d_continuous_cumsum(in_dtype="float32", out_dtype="float32", index_bits=64)
    assert_structural_equal(mod["gpu_2d_continuous_cumsum"], expected)


@pytest.mark.parametrize("index_bits", [0, 16, 128])
def test_dispatch_cumsum_invalid_index_width(index_bits):
    with pytest.raises(ValueError, match="index_bits must be either 32 or 64"):
        DispatchSortScan(index_bits=index_bits)


@pytest.mark.gpu
def test_dispatch_cumprod_cuda_large_batch():
    """Test that GPU scan supports more batches than CUDA's grid-y limit."""
    target = "cuda"
    if not device_enabled(target):
        pytest.skip(f"{target} not enabled")

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((m, n), "float32")):
            with R.dataflow():
                gv = R.cumprod(x, axis=1)
                R.output(gv)
            return gv

    np_data = np.ones((65536, 3), dtype="float32")
    np_data[:, 0] = np.arange(65536) % 7 + 1
    np_data[:, 1] = 2
    np_data[:, 2] = 3
    np_cumprod = np.cumprod(np_data, axis=1)

    mod = DispatchSortScan(target=tvm_tirx.target.Target(target))(Module)
    ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm_tirx.device_from_target(target)
        vm = tvm.relax.VirtualMachine(ex, dev)
        tvm_data = tvm_tirx.runtime.tensor(np_data, dev)
        cumprod = vm["main"](tvm_data)
        tvm_tirx.testing.assert_allclose(cumprod.numpy(), np_cumprod)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize(
    "shape, axis, in_dtype, out_dtype, expected_kernel, expected_kernel_rank",
    [
        ((3, 5), 0, "float32", None, "gpu_3d_axis_1_cumsum", 3),
        ((2, 3, 4, 5), 1, "float32", None, "gpu_3d_axis_1_cumsum", 3),
        ((2, 3, 4, 5), -2, "int32", "float32", "gpu_3d_axis_1_cumsum", 3),
        # A short scan keeps total_rounds at zero in gpu_2d_continuous_cumsum.
        ((2, 3, 4, 5), -1, "float32", None, "gpu_2d_continuous_cumsum", 2),
    ],
)
def test_dispatch_cumsum_webgpu_axes_and_dtypes(
    shape, axis, in_dtype, out_dtype, expected_kernel, expected_kernel_rank
):
    """WebGPU dispatch collapses arbitrary-rank scans to the appropriate kernel."""

    vdevice = tvm.relax.VDevice("webgpu", 0)
    x = relax.Var("x", relax.TensorType(shape, in_dtype, vdevice=vdevice))
    bb = relax.BlockBuilder()
    with bb.function("main", (x,)):
        out = bb.emit(relax.op.cumsum(x, axis=axis, dtype=out_dtype))
        bb.emit_func_output(out)
    before = bb.finalize()
    before.update_global_info("vdevice", [vdevice])

    target = tvm_tirx.target.Target("webgpu", host="llvm")
    mod = DispatchSortScan(target=target)(before)

    called_kernels = []
    permute_count = 0

    def collect_calls(expr):
        nonlocal permute_count
        if isinstance(expr, relax.Call) and getattr(expr.op, "name", None) == (
            "relax.permute_dims"
        ):
            permute_count += 1
        if isinstance(expr, relax.Call) and getattr(expr.op, "name", None) == "relax.call_tir":
            called_kernels.append(expr.args[0].name_hint)

    relax.analysis.post_order_visit(mod["main"], collect_calls)
    assert permute_count == 0
    assert called_kernels == [expected_kernel]

    cumsum = mod[expected_kernel]
    buffers = [param for param in cumsum.params if tvm_tirx.tirx.is_tensor_var(param)]
    assert len(buffers) == 2
    assert all(len(buffer.shape) == expected_kernel_rank for buffer in buffers)
    assert str(buffers[0].dtype) == in_dtype
    assert str(buffers[1].dtype) == (out_dtype or in_dtype)

    if expected_kernel == "gpu_2d_continuous_cumsum":
        floor_divisors = []

        def collect_floor_divisors(node):
            if isinstance(node, tirx.FloorDiv):
                floor_divisors.append(node.b)

        tvm_ffi.structural_walk(cumsum.body, collect_floor_divisors)
        assert floor_divisors
        assert all(
            isinstance(divisor, tvm_tirx.ir.IntImm)
            and divisor.value > 0
            and divisor.value & (divisor.value - 1) == 0
            for divisor in floor_divisors
        )

    tvm.compile(mod, target)


def test_dispatch_cumsum_webgpu_symbolic_non_contiguous_axis():
    """The serial WebGPU fallback accepts a symbolic scan extent."""

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Symbolic:
        I.module_global_infos({"vdevice": [R.vdevice("webgpu", 0)]})

        @R.function
        def main(x: R.Tensor((1, n, 9), "float32", "webgpu")):
            return R.cumsum(x, axis=1)

    target = tvm_tirx.target.Target("webgpu", host="llvm")
    mod = DispatchSortScan(target=target)(Symbolic)
    tvm.compile(mod, target)

    called_kernels = []

    def collect_calls(expr):
        if isinstance(expr, relax.Call) and getattr(expr.op, "name", None) == "relax.call_tir":
            called_kernels.append(expr.args[0].name_hint)

    relax.analysis.post_order_visit(mod["main"], collect_calls)
    assert called_kernels == ["gpu_3d_axis_1_cumsum"]


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "supports_int64": True}, marks=pytest.mark.gpu),
        pytest.param("metal", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [("float32", "float32"), ("int32", "int32"), ("int32", "float32")],
)
def test_gpu_axis_1_cumsum_numerical(target, in_dtype, out_dtype):
    """The fallback matches a sequential cumsum for supported WebGPU dtypes."""
    if not device_enabled(target):
        pytest.skip(f"{target} not enabled")

    from tvm.relax.backend.gpu_generic import (  # pylint: disable=import-outside-toplevel
        gpu_3d_axis_1_cumsum,
    )

    shape = (2, 5, 7)
    if in_dtype == "int32":
        np_data = np.random.randint(-4, 5, shape).astype(in_dtype)
    else:
        np_data = np.random.uniform(-2, 2, shape).astype(in_dtype)
    expected = np.cumsum(np_data, axis=1, dtype=out_dtype)

    func = gpu_3d_axis_1_cumsum(in_dtype=in_dtype, out_dtype=out_dtype).with_attr(
        "global_symbol", "main"
    )
    compiled = tvm.compile(func, target=target)

    def run_and_check():
        dev = tvm_tirx.device_from_target(target)
        input_tensor = tvm_tirx.runtime.tensor(np_data, dev)
        output_tensor = tvm_tirx.runtime.empty(shape, out_dtype, dev)
        compiled(input_tensor, output_tensor)
        tvm_tirx.testing.assert_allclose(output_tensor.numpy(), expected)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
