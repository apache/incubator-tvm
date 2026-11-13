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
import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def test_pipeline_compile(capsys):
    target = tvm_tirx.target.Target("llvm", host="llvm")
    pipeline = relax.pipeline.get_default_pipeline(target)

    @tvm_tirx.script.ir_module
    class Mod:
        @R.function
        def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            lv0 = R.add(x, y)
            return lv0

    mod = Mod
    from tvm_tirx.ir.transform.instrument import PrintAfterAll, PrintBeforeAll

    with tvm_tirx.ir.transform.PassContext(instruments=[PrintBeforeAll(), PrintAfterAll()]):
        mod = pipeline(mod)
    output = capsys.readouterr().out
    assert "Before Running Pass:" in output
    assert "After Running Pass:" in output
    assert 'name="_pipeline"' in output

    ex = tvm.compile(mod, target)
    x_np = np.random.rand(3, 4).astype(np.float32)
    y_np = np.random.rand(3, 4).astype(np.float32)
    x = tvm_tirx.runtime.tensor(x_np)
    y = tvm_tirx.runtime.tensor(y_np)

    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    z = vm["main"](x, y)
    tvm_tirx.testing.assert_allclose(z.numpy(), x_np + y_np, rtol=1e-7, atol=1e-7)


def test_pipeline_with_kv_cache():
    """A dummy pipline that simulates KV update."""
    target = tvm_tirx.target.Target("llvm", host="llvm")
    pipeline = relax.pipeline.get_default_pipeline(target)

    m = T.dynamic("m", "int64")
    L = T.dynamic("L", "int64")

    @tvm_tirx.script.ir_module
    class Mod:
        @R.function
        def create_kv_cache(reserve_slots: R.Shape([m])):
            # just allocate minimum slot since it is only used to signal dtype
            init_data = R.ones((1, 4), "float32")
            kv_cache = R.call_pure_packed(
                "vm.builtin.attention_kv_cache_create",
                init_data,
                R.shape([m, 4]),
                0,
                ty_args=[R.Any()],
            )
            return kv_cache

        @R.function(pure=False)
        def main(
            x: R.Tensor((1, 4), "float32"),
            y: R.Tensor((1, 4), "float32"),
            shape: R.Shape([L, 4]),
            kv_cache: R.Any,
        ):
            # computation of the current value
            curr_value = R.add(x, y)
            # update cache
            kv_cache = R.call_packed(
                "vm.builtin.attention_kv_cache_append", kv_cache, curr_value, ty_args=[R.Any]
            )
            # return the updated cache view
            kv = R.call_packed(
                "vm.builtin.attention_kv_cache_view",
                kv_cache,
                shape,
                ty_args=[R.Tensor((L, 4), "float32")],
            )
            return (kv, kv_cache)

    mod = Mod
    mod = pipeline(mod)

    ex = tvm.compile(mod, target)

    num_steps = 8
    cache_np = np.empty((num_steps, 4), dtype="float32")
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    kv_cache = vm["create_kv_cache"](tvm_ffi.Shape([1]))

    for i in range(num_steps):
        x_np = np.random.rand(1, 4).astype(np.float32)
        y_np = np.random.rand(1, 4).astype(np.float32)
        x = tvm_tirx.runtime.tensor(x_np)
        y = tvm_tirx.runtime.tensor(y_np)
        np_shape = (i + 1, 4)
        kv, kv_cache = vm["main"](x, y, tvm_ffi.Shape(np_shape), kv_cache)

        cache_np[i, :] = x_np + y_np
        tvm_tirx.testing.assert_allclose(
            kv.numpy(), cache_np[: np_shape[0], :], rtol=1e-7, atol=1e-7
        )


@pytest.mark.parametrize("target_name", ["vulkan", "webgpu"])
@pytest.mark.parametrize(
    "pipeline_func",
    [
        relax.pipeline.library_dispatch_passes,
        relax.pipeline.legalize_passes,
        relax.pipeline.dataflow_lower_passes,
        relax.pipeline.finalize_passes,
        relax.pipeline.get_default_pipeline,
    ],
)
def test_gpu_generic_fallback(target_name, pipeline_func):
    target = tvm_tirx.target.Target(target_name)
    result = pipeline_func(target)
    assert result is not None


@pytest.mark.parametrize("target_name", ["hexagon", "c"])
@pytest.mark.parametrize(
    "pipeline_func",
    [
        relax.pipeline.library_dispatch_passes,
        relax.pipeline.legalize_passes,
        relax.pipeline.dataflow_lower_passes,
        relax.pipeline.finalize_passes,
        relax.pipeline.get_default_pipeline,
    ],
)
def test_non_gpu_target_raises_error(target_name, pipeline_func):
    target = tvm_tirx.target.Target(target_name)
    with pytest.raises(ValueError, match="not yet supported"):
        pipeline_func(target)


# An elementwise binary op with a scalar constant operand. `R.power(x, const)`
# legalizes to a single elementwise TIR Function, which the default GPU pipeline
# must schedule (bind to GPU threads). Without a thread binding the kernel
# access memory from the host and `VerifyMemory` rejects it at build time
# ("... is directly accessed by the host memory ... Did you forget to bind?").
@tvm_tirx.script.ir_module
class PowerModule:
    @R.function
    def main(x: R.Tensor((1, 2, 1, 1), dtype="float32")) -> R.Tensor((1, 2, 1, 1), dtype="float32"):
        with R.dataflow():
            y: R.Tensor((1, 2, 1, 1), dtype="float32") = R.power(x, R.const(2.0, "float32"))
            R.output(y)
        return y


def _has_thread_binding(func: tvm_tirx.tirx.Function) -> bool:
    """Whether the Function body contains a GPU thread-binding loop."""
    found = False

    def _visit(node):
        nonlocal found
        if isinstance(node, tvm_tirx.ir.For) and node.annotations.get("thread_binding") is not None:
            found = True

    tvm_ffi.structural_walk(func.body, _visit)
    return found


def test_default_cuda_pipeline_schedules_power():
    """Exercise CUDA pipeline selection and verify compute kernel is thread-bound.

    Device-free test (no GPU required) that exercises the backend-level pipeline
    selection path (get_default_pipeline) for CUDA targets and verifies that
    legalized elementwise compute kernels receive GPU thread bindings.

    Note: The full pipeline selection routing in vm_build.py (Layer 1: "gpu" in
    target.keys decision) is covered by the GPU-gated end-to-end test
    (test_power_cuda_build_and_run). This test focuses on Layer 2: backend-specific
    pipeline composition (DLight scheduling application).
    """
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_86"})

    # Verify: Target must be recognized as GPU (Layer 1 prerequisite)
    assert "gpu" in target.keys, "CUDA target must be marked as GPU for proper pipeline selection"

    # Exercise: Get the backend-specific default pipeline for this target
    pipeline = relax.pipeline.get_default_pipeline(target)
    mod = pipeline(PowerModule)

    # Verify: At least one compute kernel has thread binding.
    # (Using any() avoids false-fail from host-side shape helper kernels
    # that may be generated by finalize_passes but don't need GPU binding)
    functions = [
        func for _, func in mod.functions.items() if isinstance(func, tvm_tirx.tirx.Function)
    ]
    assert functions, "expected at least one TIR Function after pipeline"

    has_bound_kernel = any(_has_thread_binding(f) for f in functions)
    assert has_bound_kernel, (
        "At least one compute kernel must have GPU thread binding; "
        "shape helper kernels may not be bound"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_power_cuda_build_and_run():
    """End-to-End build and run of an elementwise `R.power` kernel on CUDA.

    Compiles through `tvm.compile`, which for a GPU target selects the
    target-specific default pipeline (with DLight scheduling), then executes the
    kernel and checks the result.
    """
    dev = tvm_tirx.cuda(0)
    target = tvm_tirx.target.Target.from_device(dev)

    ex = tvm.compile(PowerModule, target=target)
    vm = relax.VirtualMachine(ex, dev)

    x_np = np.random.rand(1, 2, 1, 1).astype(np.float32)
    out = vm["main"](tvm_tirx.runtime.tensor(x_np, dev))
    tvm_tirx.testing.assert_allclose(out.numpy(), x_np**2, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tvm_tirx.testing.main()
