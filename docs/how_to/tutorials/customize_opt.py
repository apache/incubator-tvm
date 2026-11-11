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
# ruff: noqa: E402, E501, F401

"""
.. _customize_opt:

Customize Optimization
======================
One main design goal of Apache TVM is to enable easy customization of the optimization pipeline
for both research or development purposes and iterate the engineering optimizations. In this
tutorial we will

.. contents:: Table of Contents
    :local:
    :depth: 1
"""

######################################################################
# Review Overall Flow
# -------------------
# .. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/design/tvm_overall_flow.svg
#    :align: center
#    :width: 80%
#
# The overall flow consists of the following steps:
#
# - **Construct or Import a Model**: Construct a neural network model or import a pre-trained
#   model from other frameworks (e.g. PyTorch, ONNX), and create the TVM IRModule, which contains
#   all the information needed for compilation, including high-level Relax functions for
#   computational graph, and low-level TensorIR functions for tensor program.
# - **Perform Composable Optimizations**: Perform a series of optimization transformations,
#   such as graph optimizations, tensor program optimizations, and library dispatching.
# - **Build and Universal Deployment**: Build the optimized model to a deployable module to the
#   universal runtime, and execute it on different devices, such as CPU, GPU, or other accelerators.
#
import os
import tempfile

import numpy as np
import tvm_ffi

import tvm
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm_tirx import IRModule

######################################################################
# Composable IRModule Optimization
# --------------------------------
# Apache TVM provides a flexible way to optimize the IRModule. Everything centered
# around IRModule optimization can be composed with existing pipelines. Note that each optimization
# can focus on **part of the computation graph**, enabling partial lowering or partial optimization.
#
# In this tutorial, we will demonstrate how to optimize a model with Apache TVM.

######################################################################
# Prepare a Relax Module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# We first prepare a Relax module. The module can be imported from other frameworks, constructed
# with NN module frontend or TVMScript. Here we use a simple neural network model as an example.


class RelaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


input_shape = (1, 784)
mod, params = RelaxModel().export_tvm({"forward": {"x": nn.spec.Tensor(input_shape, "float32")}})
mod.show()

######################################################################
# Library Dispatch
# ~~~~~~~~~~~~~~~~
# We would like to quickly try out a variant of library optimization for certain platforms
# (e.g., GPU). We can write a certain dispatching pass for the specific platform and
# operator. Here we demonstrate how to dispatch the CUBLAS library for certain patterns.
#
# .. note::
#   This tutorial only demonstrates a single operator dispatching for CUBLAS, highlighting
#   the flexibility of the optimization pipeline. In real-world cases, we can import multiple
#   patterns and dispatch them to different kernels.


# Import cublas pattern
try:
    import tvm.relax.backend.cuda.cublas as _cublas
except ImportError as e:
    raise ImportError(
        "This tutorial requires TVM built with CUDA support.\n"
        "If you hit missing 'tvm_ffi', try: pip install apache-tvm-ffi\n"
        "Otherwise build TVM with CUDA enabled:\n"
        "  https://tvm.apache.org/docs/install/from_source.html\n"
        f"Original error: {e}"
    ) from e


# Define a new pass for CUBLAS dispatch
@tvm_tirx.ir.transform.module_pass(opt_level=0, name="CublasDispatch")
class CublasDispatch:
    def transform_module(self, mod: IRModule, _ctx: tvm_tirx.ir.transform.PassContext) -> IRModule:
        # Check if CUBLAS is enabled
        if not tvm_ffi.get_global_func("relax.ext.cublas", True):
            raise Exception("CUBLAS is not enabled.")

        # Get interested patterns
        patterns = [relax.backend.get_pattern("cublas.matmul_transposed_bias_relu")]
        # Note in real-world cases, we usually get all patterns
        # patterns = relax.backend.get_patterns_with_prefix("cublas")

        # Fuse ops by patterns and then run codegen
        mod = relax.transform.FuseOpsByPattern(patterns, annotate_codegen=True)(mod)
        mod = relax.transform.RunCodegen()(mod)
        return mod


mod = CublasDispatch()(mod)
mod.show()

######################################################################
# After the dispatching pass, we can see that the first ``nn.Linear`` and ``nn.ReLU`` are fused
# and rewritten to a ``call_dps_packed`` function which call the CUBLAS library. Notably, the
# other part is not changed, which means we can selectively dispatch the optimization for
# certain computation.

######################################################################
# Auto Tuning
# ~~~~~~~~~~~
# Continuing from the previous example, we can further optimize the model with auto-tuning for
# the **rest part of the computation**. Here we demonstrate how to use the meta-schedule to auto-tune
# the model.
#
# We can use ``MetaScheduleTuneTIR`` pass to simply tuning the model, while ``MetaScheduleApplyDatabase``
# pass to apply the best configuration to the model. The tuning process will generate search space,
# tune the model and the following steps will apply the best configuration to the model. Before
# running the passes, we need to lowering relax operator into TensorIR functions via ``LegalizeOps``
#
# .. note::
#
#   To save CI time and avoid flakiness, we skip the tuning process in CI environment.
#

device = tvm_tirx.cuda(0)
target = tvm_tirx.target.Target.from_device(device)
if os.getenv("CI", "") != "true":
    trials = 2000
    with tempfile.TemporaryDirectory() as tmp_dir:
        mod = tvm_tirx.ir.transform.Sequential(
            [
                relax.get_pipeline("zero", target=target),
                relax.transform.MetaScheduleTuneTIR(
                    work_dir=tmp_dir, max_trials_global=trials, target=target
                ),
                relax.transform.MetaScheduleApplyDatabase(work_dir=tmp_dir, target=target),
            ]
        )(mod)

    mod.show()

######################################################################
# DLight Rules
# ~~~~~~~~~~~~
# DLight rules are a set of default rules for scheduling and optimization the kernel.
# DLight rules are designed for fast compilation and **fair** performance. In some cases,
# e.g. language model, DLight provides excellent performance, while for generic models,
# it achieves a balance between performance and compilation time.

from tvm.s_tir import dlight as dl

# Apply DLight rules
mod = tvm_tirx.ir.transform.Sequential(
    [
        relax.get_pipeline("zero", target=target),
        dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
            target=target,
        ),
    ]
)(mod)

mod.show()

######################################################################
# .. note::
#
#   This tutorial focuses on the demonstration of the optimization pipeline, instead of
#   pushing the performance to the limit. The current optimization may not be the best.


######################################################################
# Deploy the Optimized Model
# --------------------------
# We can build and deploy the optimized model to the TVM runtime.

ex = tvm.compile(mod, target="cuda")
dev = tvm_tirx.cuda(0)
vm = relax.VirtualMachine(ex, dev)
# Need to allocate data and params on GPU device
data = tvm_tirx.runtime.tensor(np.random.rand(*input_shape).astype("float32"), dev)
gpu_params = [
    tvm_tirx.runtime.tensor(np.random.rand(*p.shape).astype(p.dtype), dev) for _, p in params
]
gpu_out = vm["forward"](data, *gpu_params).numpy()
print(gpu_out)


######################################################################
# Summary
# -------
# This tutorial demonstrates how to customize the optimization pipeline for ML models in Apache TVM.
# We can easily compose the optimization passes and customize the optimization for different parts
# of the computation graph. The flexibility of the optimization pipeline enables us to quickly
# iterate the optimization and improve the performance of the model.
#
