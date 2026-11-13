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

"""
.. _tutorial-bring-your-own-codegen:

Bring Your Own Codegen
======================

TVM's Bring Your Own Codegen (BYOC) framework lets you offload parts of a model
to a custom backend -- a hardware accelerator, an inference library, or your own
kernels -- while TVM compiles the rest.  This tutorial has two parts:

- **How BYOC works** -- we teach the flow with a bundled, hardware-free *example
  NPU* backend and then drive the **same flow** on a real production backend,
  NVIDIA TensorRT.  Both run a small, hand-written model so every step is
  visible; the only thing that changes between them is the backend, and that
  contrast is the lesson.
- **Deploying a real model** -- we then put it to work, taking an actual PyTorch
  ``nn.Module`` from export through TensorRT and running it on the GPU.

The example NPU is a teaching stub: its runtime logs the dispatch decisions an
NPU would make (memory tier, execution engine, fusion) but performs no real
computation, so its output buffers are left uninitialized.  We therefore check
*shapes*, not values, in the NPU sections -- its job is to make every BYOC step
visible with nothing hidden.  TensorRT then runs the identical flow for real, so
we cross-check its result against a reference.

**Prerequisites**: the example NPU sections need TVM built with
``TVM_USE_EXAMPLE_NPU_CODEGEN=ON`` and ``TVM_USE_EXAMPLE_NPU_RUNTIME=ON``; the TensorRT
sections need ``TVM_USE_TENSORRT_CODEGEN=ON``, ``TVM_USE_TENSORRT_RUNTIME=ON`` and
``TVM_TIRX_USE_CUDA=ON`` plus a CUDA GPU and a matching TensorRT install (from NVIDIA's
``pip install tensorrt`` packages or the TensorRT archive); the final deployment
section also needs PyTorch.  Each section degrades gracefully when its backend is
unavailable.
"""

######################################################################
# Overview of the BYOC flow
# -------------------------
#
# BYOC plugs a custom backend into TVM's compilation pipeline in four steps:
#
# 1. **Register patterns** - describe which sequences of Relax ops the backend
#    can handle.
# 2. **Partition the graph** - group matched ops into composite functions.
# 3. **Run codegen** - lower each composite to the backend's representation
#    (a JSON graph for both backends in this tutorial).
# 4. **Execute** - the runtime dispatches each composite to the backend.
#
# Steps 1 and 2 are pure Python and run anywhere; steps 3 and 4 need the
# backend's codegen and runtime compiled into TVM, which is why the
# build-and-run cells below are guarded.
#
# Step 1: Import the backends to register their patterns
# ------------------------------------------------------
#
# Importing a backend module registers its patterns with TVM's global registry.
# Pattern registration is independent of the C++ build -- only codegen and the
# runtime require the backend to be compiled in -- so we probe each backend and
# guard the build-and-run cells accordingly.
import os
import tempfile

import numpy as np
import tvm_ffi

import tvm.relax.backend.contrib.example_npu  # noqa: F401
import tvm_tirx
from tvm import relax
from tvm.relax import script as R
from tvm.relax.backend.contrib.tensorrt import partition_for_tensorrt
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import FuseOpsByPattern, MergeCompositeFunctions, RunCodegen

has_example_npu_codegen = tvm_ffi.get_global_func("relax.ext.example_npu", True)
has_example_npu_runtime = tvm_ffi.get_global_func("runtime.ExampleNPUJSONRuntimeCreate", True)
has_example_npu = has_example_npu_codegen and has_example_npu_runtime

has_tensorrt_codegen = tvm_ffi.get_global_func("relax.ext.tensorrt", True) is not None
_is_trt_runtime_enabled = tvm_ffi.get_global_func("relax.is_tensorrt_runtime_enabled", True)
has_tensorrt = (
    has_tensorrt_codegen and _is_trt_runtime_enabled is not None and _is_trt_runtime_enabled()
)
has_cuda = tvm_tirx.cuda(0).exist

######################################################################
# Step 2: Define the model
# ------------------------
#
# A single convolution followed by a ReLU.  This one model is used for both
# backends.


@tvm_tirx.script.ir_module
class ConvReLU:
    @R.function
    def main(
        data: R.Tensor((1, 3, 32, 32), "float32"),
        weight: R.Tensor((16, 3, 3, 3), "float32"),
    ) -> R.Tensor((1, 16, 30, 30), "float32"):
        with R.dataflow():
            conv = relax.op.nn.conv2d(data, weight)
            out = relax.op.nn.relu(conv)
            R.output(out)
        return out


######################################################################
# Step 3: Partition for the example NPU
# -------------------------------------
#
# ``FuseOpsByPattern`` groups ops matching a registered pattern into composite
# functions; ``MergeCompositeFunctions`` then consolidates adjacent composites
# bound for the same backend into a single external call.  Two flags steer
# partitioning:
#
# - ``bind_constants=False`` keeps weights as function arguments, so the host
#   stays in charge of the parameters.  (TensorRT below makes the opposite
#   choice: it binds weights as constants because it bakes them into its engine.)
# - ``annotate_codegen=True`` wraps each matched composite in a function tagged
#   with the backend name -- the tag ``RunCodegen`` routes on.  (The follow-up
#   ``MergeCompositeFunctions`` also attaches this tag when it groups composites,
#   which is why ``partition_for_tensorrt`` below can leave the flag off.)
#
# The example NPU registers a fused ``conv2d + relu`` pattern with higher
# priority than the standalone ``conv2d`` pattern, so the two ops collapse into a
# single ``example_npu.conv2d_relu_fused`` composite -- look for it in the
# printed module.

npu_patterns = get_patterns_with_prefix("example_npu")
npu_mod = FuseOpsByPattern(npu_patterns, bind_constants=False, annotate_codegen=True)(ConvReLU)
npu_mod = MergeCompositeFunctions()(npu_mod)
print("After partitioning for the example NPU:")
print(npu_mod)

######################################################################
# Step 4: Codegen, build and run on the example NPU
# -------------------------------------------------
#
# ``RunCodegen`` invokes each annotated composite's backend codegen, replacing it
# with the backend runtime module (here, the NPU's JSON graph); ``relax.build``
# then compiles the remaining host-side program and links everything.  Because
# the runtime is a stub that computes nothing, we assert on the output *shape*
# only -- the values are uninitialized.

np.random.seed(0)
data_np = np.random.randn(1, 3, 32, 32).astype("float32")
weight_np = np.random.randn(16, 3, 3, 3).astype("float32")

if has_example_npu:
    npu_mod = RunCodegen()(npu_mod)

    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        npu_exec = relax.build(npu_mod, tvm_tirx.target.Target("llvm"))

    npu_vm = relax.VirtualMachine(npu_exec, tvm_tirx.cpu())
    npu_out = npu_vm["main"](
        tvm_tirx.runtime.tensor(data_np, tvm_tirx.cpu()),
        tvm_tirx.runtime.tensor(weight_np, tvm_tirx.cpu()),
    )
    assert npu_out.numpy().shape == (1, 16, 30, 30)
    print("Example NPU run completed. Output shape:", npu_out.numpy().shape)
else:
    print("Example NPU backend unavailable; skipping its build and run.")

######################################################################
# The same flow on a real backend: TensorRT
# -----------------------------------------
#
# Steps 1-4 above are the whole mechanism.  Aiming them at a real backend
# changes very little, so rather than repeat the walkthrough, here is only what
# differs for NVIDIA TensorRT:
#
# - **Partition in one call.** ``partition_for_tensorrt`` bundles the
#   ``FuseOpsByPattern`` + ``MergeCompositeFunctions`` you ran by hand, using
#   TensorRT's own pattern table.
# - **Weights become constants** (``bind_constants=True``): TensorRT bakes them
#   into the engine it builds, so bind the parameters before partitioning.
# - **Real values.** TensorRT actually computes, so we build for CUDA, run on
#   the GPU, and cross-check against a plain CPU build -- not just the shape.
#
# The build-and-run cells below execute only when TensorRT and CUDA are
# available. In CPU-only documentation builds, they produce no output.

trt_mod = relax.transform.BindParams("main", {"weight": weight_np})(ConvReLU)
trt_mod = partition_for_tensorrt(trt_mod)
print("After partition_for_tensorrt:")
print(trt_mod)

######################################################################
# Build for CUDA, run on the GPU, and compare against the CPU reference.

if has_tensorrt and has_cuda:
    dev = tvm_tirx.cuda(0)
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        trt_exec = relax.build(RunCodegen()(trt_mod), "cuda")
    trt_out = relax.VirtualMachine(trt_exec, dev)["main"](
        tvm_tirx.runtime.tensor(data_np, dev)
    ).numpy()

    cpu_mod = relax.transform.LegalizeOps()(
        relax.transform.BindParams("main", {"weight": weight_np})(ConvReLU)
    )
    cpu_exec = relax.build(cpu_mod, "llvm")
    cpu_out = relax.VirtualMachine(cpu_exec, tvm_tirx.cpu())["main"](
        tvm_tirx.runtime.tensor(data_np, tvm_tirx.cpu())
    ).numpy()

    np.testing.assert_allclose(trt_out, cpu_out, rtol=1e-2, atol=1e-2)
    print("TensorRT output shape:", trt_out.shape, "- matches the CPU reference.")

######################################################################
# A real backend also exposes knobs the stub does not.  Setting ``use_fp16``
# through the ``relax.ext.tensorrt.options`` config lets TensorRT pick FP16
# kernels, trading a little accuracy for speed; nothing else about the flow
# changes.  (Other options are environment-driven: ``TVM_TENSORRT_USE_INT8``
# enables INT8 with calibration, ``TVM_TENSORRT_MAX_WORKSPACE_SIZE`` caps the
# build workspace, and ``TVM_TENSORRT_CACHE_DIR`` caches built engines to disk
# for reuse across runs.)

if has_tensorrt and has_cuda:
    fp16_mod = partition_for_tensorrt(
        relax.transform.BindParams("main", {"weight": weight_np})(ConvReLU)
    )
    with tvm_tirx.ir.transform.PassContext(
        opt_level=3, config={"relax.ext.tensorrt.options": {"use_fp16": True}}
    ):
        fp16_exec = relax.build(RunCodegen()(fp16_mod), "cuda")
    fp16_out = relax.VirtualMachine(fp16_exec, tvm_tirx.cuda(0))["main"](
        tvm_tirx.runtime.tensor(data_np, tvm_tirx.cuda(0))
    ).numpy()

    np.testing.assert_allclose(fp16_out, cpu_out, rtol=5e-2, atol=5e-2)
    print("TensorRT FP16 output shape:", fp16_out.shape, "- matches within FP16 tolerance.")

######################################################################
# Example NPU vs TensorRT at a glance
# -----------------------------------
#
# The same four-step flow, two backends:
#
# =========  ==============================  ==================================
# Aspect     Example NPU (teaching stub)     TensorRT (real backend)
# =========  ==============================  ==================================
# Runtime    logs decisions, no compute      builds and runs an nvinfer engine
# Output     uninitialized (check shape)     real values (cross-checked vs CPU)
# Weights    ``bind_constants=False``        ``bind_constants=True`` (baked in)
# Partition  two passes, by hand             ``partition_for_tensorrt`` one call
# =========  ==============================  ==================================

######################################################################
# Deploying a PyTorch model with TensorRT
# ---------------------------------------
#
# Everything above used a hand-written ``IRModule`` so each op was visible.  In
# practice you start from a trained model.  This final section runs the *same*
# ``partition_for_tensorrt`` flow on a real PyTorch ``nn.Module``, end to end:
# export it, import it into Relax with the PyTorch frontend (the weights come in
# as constants -- exactly what TensorRT bakes into its engine), partition, build
# for CUDA, and check the GPU result against PyTorch's own output.  Beyond the
# frontend import, the only difference is that the imported program returns its
# outputs as a tuple, so we index ``[0]`` for the single result tensor; the
# partition-build-run flow is otherwise unchanged.
#
# This section additionally requires PyTorch.

try:
    import torch
    from torch import nn

    has_torch = True
except ImportError:
    has_torch = False

if has_torch and has_tensorrt and has_cuda:
    from tvm.relax.frontend.torch import from_exported_program

    class SmallConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3)
            self.conv2 = nn.Conv2d(8, 16, 3)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            return x

    torch_model = SmallConvNet().eval()
    example_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        torch_ref = torch_model(example_input).numpy()
        exported = torch.export.export(torch_model, (example_input,))

    torch_mod = from_exported_program(exported)
    torch_mod = partition_for_tensorrt(torch_mod)
    print("After importing and partitioning the PyTorch model:")
    print(torch_mod)

    torch_dev = tvm_tirx.cuda(0)
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        torch_exec = relax.build(RunCodegen()(torch_mod), "cuda")
    deployed = relax.VirtualMachine(torch_exec, torch_dev)["main"](
        tvm_tirx.runtime.tensor(example_input.numpy(), torch_dev)
    )[0].numpy()

    np.testing.assert_allclose(deployed, torch_ref, rtol=1e-2, atol=1e-2)
    print("Deployed PyTorch model on TensorRT; output", deployed.shape, "matches PyTorch.")

######################################################################
# Real deployment builds once and reuses the artifact.  Export the compiled
# module to a shared library, then load and run it later -- in a fresh process,
# with no PyTorch and no rebuild needed.

if has_torch and has_tensorrt and has_cuda:
    with tempfile.TemporaryDirectory() as tmpdir:
        lib_path = os.path.join(tmpdir, "deployed_trt.so")
        torch_exec.export_library(lib_path)
        loaded = tvm_tirx.runtime.load_module(lib_path)
        reran = relax.VirtualMachine(loaded, torch_dev)["main"](
            tvm_tirx.runtime.tensor(example_input.numpy(), torch_dev)
        )[0].numpy()
        np.testing.assert_allclose(reran, torch_ref, rtol=1e-2, atol=1e-2)
        print("Reloaded the exported library and reran; output", reran.shape, "still matches.")

######################################################################
# Notes for real deployments
# --------------------------
#
# - **Operator coverage and fallback.** TensorRT offloads only the ops in its
#   pattern table (see ``python/tvm/relax/backend/contrib/tensorrt.py``);
#   anything unsupported simply stays on the host.  Print the partitioned module
#   and look for the ``Codegen: "tensorrt"`` functions to see what was offloaded.
# - **Dynamic shapes.** The builder sets up an optimization profile for a dynamic
#   leading (batch) dimension, so the integration can serve a model exported with
#   a symbolic batch size.
# - **Engine build cost.** Building a TensorRT engine is slow the first time (it
#   is not a hang).  Set ``TVM_TENSORRT_CACHE_DIR`` to cache built engines to
#   disk and skip the rebuild on later runs.

######################################################################
# Next steps
# ----------
#
# To build your own backend using the example NPU as a starting point:
#
# - Replace the stub runtime in
#   ``src/runtime/extra/contrib/example_npu/example_npu_runtime.cc`` with your
#   hardware SDK calls.
# - Extend ``patterns.py`` with the ops your hardware supports.
# - Add a C++ codegen under ``src/relax/backend/contrib/`` if your backend needs
#   a non-JSON serialization format.
# - Add a CMake module under ``cmake/modules/contrib/`` following
#   ``ExampleNPU.cmake``.
#
# For a complete real-backend implementation to study, see the TensorRT
# integration: the pattern table and ``partition_for_tensorrt`` in
# ``python/tvm/relax/backend/contrib/tensorrt.py``, the codegen in
# ``src/relax/backend/contrib/tensorrt/``, and the runtime in
# ``src/runtime/extra/contrib/tensorrt/``.
