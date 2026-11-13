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
# ruff: noqa: E402

"""
.. _deploy_export_and_load_executable:

Export and Load Relax Executables
=================================

This tutorial walks through exporting a compiled Relax module to a shared
object, loading it back into the TVM runtime, and running the result either
interactively or from a standalone script. This tutorial demonstrates how
to turn Relax (or imported PyTorch / ONNX) programs into deployable artifacts
using ``tvm.relax`` APIs.

.. note::
   This tutorial uses PyTorch as the source format, but the export/load workflow
   is the same for ONNX models. For ONNX, use ``from_onnx(model, keep_params_in_input=True)``
   instead of ``from_exported_program()``, then follow the same steps for building,
   exporting, and loading.
"""

######################################################################
# Introduction
# ------------
# TVM builds Relax programs into ``tvm_tirx.codegen.Executable`` objects. These
# contain VM bytecode, compiled kernels, and constants. By exporting the
# executable with :py:meth:`export_library`, you obtain a shared library (for
# example ``.so`` on Linux) that can be shipped to another machine, uploaded
# via RPC, or loaded back later with the TVM runtime. This tutorial shows the
# exact steps end-to-end and explains what files are produced along the way.

import os
from pathlib import Path

try:
    import torch
    from torch.export import export
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


######################################################################
# Prepare a Torch MLP and Convert to Relax
# ----------------------------------------
# We start with a small PyTorch MLP so the example remains lightweight. The
# model is exported to a :py:class:`torch.export.ExportedProgram` and then
# translated into a Relax ``IRModule``.

import tvm_ffi

import tvm
import tvm_tirx
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Check dependencies first
IS_IN_CI = os.getenv("CI", "").lower() == "true"
HAS_TORCH = torch is not None
RUN_EXAMPLE = HAS_TORCH and not IS_IN_CI


if HAS_TORCH:

    class TorchMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28 * 28, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10),
            )

        def forward(self, data: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(data)

else:  # pragma: no cover
    TorchMLP = None  # type: ignore[misc, assignment]

if RUN_EXAMPLE:
    torch_model = TorchMLP().eval()
    example_args = (torch.randn(1, 1, 28, 28, dtype=torch.float32),)

    with torch.no_grad():
        exported_program = export(torch_model, example_args)

    mod = from_exported_program(exported_program, keep_params_as_input=True)

    # Separate model parameters so they can be bound later (or stored on disk).
    mod, params = relax.frontend.detach_params(mod)

    print("Imported Relax module:")
    mod.show()


######################################################################
# Build and Export with ``export_library``
# -------------------------------------------
# We build for ``llvm`` to generate CPU code and then export the resulting
# executable. Passing ``workspace_dir`` keeps the intermediate packaging files,
# which is useful to inspect what was produced.

TARGET = tvm_tirx.target.Target("llvm")
ARTIFACT_DIR = Path("relax_export_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

if RUN_EXAMPLE:
    # Apply the default Relax compilation pipeline before building.
    pipeline = relax.get_pipeline()
    with TARGET:
        built_mod = pipeline(mod)

    # Build without params - we'll pass them at runtime
    executable = tvm.compile(built_mod, target=TARGET)

    library_path = ARTIFACT_DIR / "mlp_cpu.so"
    executable.export_library(str(library_path), workspace_dir=str(ARTIFACT_DIR))

    print(f"Exported runtime library to: {library_path}")

    # The workspace directory now contains the shared object and supporting files.
    produced_files = sorted(p.name for p in ARTIFACT_DIR.iterdir())
    print("Artifacts saved:")
    for name in produced_files:
        print(f"  - {name}")

    # Generated files:
    #   - ``mlp_cpu.so``: The main deployable shared library containing VM bytecode,
    #     compiled kernels, and constants. Note: Since parameters are passed at runtime,
    #     you will also need to save a separate parameters file (see next section).
    #   - Intermediate object files (``devc.o``, ``lib0.o``, etc.) are kept in the
    #     workspace for inspection but are not required for deployment.
    #
    #   Note: Additional files like ``*.params``, ``*.metadata.json``, or ``*.imports``
    #   may appear in specific configurations but are typically embedded into the
    #   shared library or only generated when needed.


######################################################################
# Load the Exported Library and Run It
# ------------------------------------
# Once the shared object is produced, we can reload it back into the TVM runtime
# on any machine with a compatible instruction set. The Relax VM consumes the
# runtime module directly.

if RUN_EXAMPLE:
    loaded_rt_mod = tvm_tirx.runtime.load_module(str(library_path))
    dev = tvm_tirx.cpu(0)
    vm = relax.VirtualMachine(loaded_rt_mod, dev)

    # Prepare input data
    input_tensor = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    vm_input = tvm_tirx.runtime.tensor(input_tensor.numpy(), dev)

    # Prepare parameters (allocate on target device)
    vm_params = [tvm_tirx.runtime.tensor(p, dev) for p in params["main"]]

    # Run inference: pass input data followed by all parameters
    tvm_output = vm["main"](vm_input, *vm_params)

    # TVM returns Array objects for tuple outputs, access via indexing.
    # For models imported from PyTorch, outputs are typically tuples (even for single outputs).
    # For ONNX models, outputs may be a single Tensor directly.
    if isinstance(tvm_output, tvm_ffi.Array) and len(tvm_output) > 0:
        result_tensor = tvm_output[0]
    else:
        result_tensor = tvm_output

    print("VM output shape:", result_tensor.shape)
    print("VM output type:", type(tvm_output), "->", type(result_tensor))

    # You can still inspect the executable after reloading.
    print("Executable stats:\n", loaded_rt_mod["stats"]())


######################################################################
# Save Parameters for Deployment
# -------------------------------
# Since parameters are passed at runtime (not embedded in the ``.so``), we must
# save them separately for deployment. This is a required step to use the model
# on other machines or in standalone scripts.

import numpy as np

if RUN_EXAMPLE:
    # Save parameters to disk
    params_path = ARTIFACT_DIR / "model_params.npz"
    param_arrays = {f"p_{i}": p.numpy() for i, p in enumerate(params["main"])}
    np.savez(str(params_path), **param_arrays)
    print(f"Saved parameters to: {params_path}")

# Note: Alternatively, you can embed parameters directly into the ``.so`` to
# create a single-file deployment. Use ``keep_params_as_input=False`` when
# importing from PyTorch:
#
# .. code-block:: python
#
#    mod = from_exported_program(exported_program, keep_params_as_input=False)
#    # Parameters are now embedded as constants in the module
#    executable = tvm.compile(built_mod, target=TARGET)
#    # Runtime: vm["main"](input)  # No need to pass params!
#
# This creates a single-file deployment (only the ``.so`` is needed), but you
# lose the flexibility to swap parameters without recompiling. For most
# production workflows, separating code and parameters (as shown above) is
# preferred for flexibility.


######################################################################
# Loading and Running the Exported Model
# -----------------------------------------------------------
# To use the exported model on another machine or in a standalone script, you need
# to load both the ``.so`` library and the parameters file. Here's a complete example
# of how to reload and run the model. Save this as ``run_mlp.py``:
#
# To make it executable from the command line:
#
# .. code-block:: bash
#
#    chmod +x run_mlp.py
#    ./run_mlp.py  # Run it like a regular program
#
# Complete script:
#
# .. code-block:: python
#
#    #!/usr/bin/env python3
#    import numpy as np
#    import tvm
#    from tvm import relax
#
#    # Step 1: Load the compiled library
#    lib = tvm.runtime.load_module("relax_export_artifacts/mlp_cpu.so")
#
#    # Step 2: Create Virtual Machine
#    device = tvm.cpu(0)
#    vm = relax.VirtualMachine(lib, device)
#
#    # Step 3: Load parameters from the .npz file
#    params_npz = np.load("relax_export_artifacts/model_params.npz")
#    params = [tvm.runtime.tensor(params_npz[f"p_{i}"], device)
#              for i in range(len(params_npz))]
#
#    # Step 4: Prepare input data
#    data = np.random.randn(1, 1, 28, 28).astype("float32")
#    input_tensor = tvm.runtime.tensor(data, device)
#
#    # Step 5: Run inference (pass input followed by all parameters)
#    output = vm["main"](input_tensor, *params)
#
#    # Step 6: Extract result (output may be tuple or single Tensor)
#    # PyTorch models typically return tuples, ONNX models may return a single Tensor
#    if isinstance(output, tvm_ffi.Array) and len(output) > 0:
#        result_tensor = output[0]
#    else:
#        result_tensor = output
#
#    print("Prediction shape:", result_tensor.shape)
#    print("Predicted class:", np.argmax(result_tensor.numpy()))
#
# **Running on GPU:**
# To run on GPU instead of CPU, make the following changes:
#
# 1. **Compile for GPU** (earlier in the tutorial, around line 112):
#
#    .. code-block:: python
#
#       TARGET = tvm.target.Target("cuda")  # Change from "llvm" to "cuda"
#
# 2. **Use GPU device in the script**:
#
#    .. code-block:: python
#
#       device = tvm.cuda(0)  # Use CUDA device instead of CPU
#       vm = relax.VirtualMachine(lib, device)
#
#       # Load parameters to GPU
#       params = [tvm.runtime.tensor(params_npz[f"p_{i}"], device)  # Note: device parameter
#                 for i in range(len(params_npz))]
#
#       # Prepare input on GPU
#       input_tensor = tvm.runtime.tensor(data, device)  # Note: device parameter
#
#    The rest of the script remains the same. All tensors (parameters and inputs)
#    must be allocated on the same device (GPU) as the compiled model.
#
# **Deployment Checklist:**
# When moving to another host (via RPC or SCP), you must copy **both** files:
#
# 1. ``mlp_cpu.so`` (or ``mlp_cuda.so`` for GPU) - the compiled model code
# 2. ``model_params.npz`` - the model parameters, serialized as NumPy arrays
#
# The remote machine needs both files in the same directory. The script above
# assumes they are in ``relax_export_artifacts/`` relative to the script location.
# Adjust the paths as needed for your deployment. For GPU deployment, ensure the
# target machine has compatible CUDA drivers and the model was compiled for the
# same GPU architecture.


######################################################################
# Deploying to Remote Devices
# ---------------------------
# To deploy the exported model to a remote ARM Linux device (e.g., Raspberry Pi),
# you can use TVM's RPC mechanism to cross-compile, upload, and run the model
# remotely. This workflow is useful when:
#
# - The target device has limited resources for compilation
# - You want to fine-tune performance by running on the actual hardware
# - You need to deploy to embedded devices
#
# See :doc:`cross_compilation_and_rpc </how_to/tutorials/cross_compilation_and_rpc>`
# for a comprehensive guide on:
#
# - Setting up TVM runtime on the remote device
# - Starting an RPC server on the device
# - Cross-compiling for ARM targets (e.g., ``llvm -mtriple=aarch64-linux-gnu``)
# - Uploading exported libraries via RPC
# - Running inference remotely
#
# Quick example for ARM deployment workflow:
#
# .. code-block:: python
#
#    import tvm.rpc as rpc
#    from tvm import relax
#
#    # Step 1: Cross-compile for ARM target (on local machine)
#    TARGET = tvm.target.Target({"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
#    executable = tvm.compile(built_mod, target=TARGET)
#    executable.export_library("mlp_arm.so")
#
#    # Step 2: Connect to remote device RPC server
#    remote = rpc.connect("192.168.1.100", 9090)  # Device IP and RPC port
#
#    # Step 3: Upload the compiled library and parameters
#    remote.upload("mlp_arm.so")
#    remote.upload("model_params.npz")
#
#    # Step 4: Load and run on remote device
#    lib = remote.load_module("mlp_arm.so")
#    vm = rpc.virtual_machine(lib, remote.cpu())
#    # ... prepare input and params, then run inference
#
# The key difference is using an ARM target triple during compilation and
# uploading files via RPC instead of copying them directly.


######################################################################
# FAQ
# ---
# **Can I run the ``.so`` as a standalone executable (like ``./mlp_cpu.so``)?**
# No. The ``.so`` file is a shared library, not a standalone executable binary.
# You cannot run it directly from the terminal. It must be loaded through a TVM
# runtime program (as shown in the "Loading and Running" section above). The
# ``.so`` bundles VM bytecode and compiled kernels, but still requires the TVM
# runtime to execute.
#
# **Which devices can run the exported library?**
# The target must match the ISA you compiled for (``llvm`` in this example).
# As long as the target triple, runtime ABI, and available devices line up,
# you can move the artifact between machines. For heterogeneous builds (CPU
# plus GPU), ship the extra device libraries as well.
#
# **What about the ``.params`` and ``metadata.json`` files?**
# These auxiliary files are only generated in specific configurations. In this
# tutorial, since we pass parameters at runtime, they are not generated. When
# they do appear, they may be kept alongside the ``.so`` for inspection, but
# the essential content is typically embedded in the shared object itself, so
# deploying the ``.so`` alone is usually sufficient.
