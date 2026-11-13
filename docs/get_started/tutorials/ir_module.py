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
.. _ir_module:

IRModule
========
This tutorial presents the core abstraction of Apache TVM, the IRModule.
The IRModule encompasses the **entirety** of the ML models, incorporating the
computational graph, tensor programs, and potential calls to external libraries.

.. contents:: Table of Contents
    :local:
    :depth: 1
"""

import numpy as np

######################################################################
# Create IRModule
# ---------------
# IRModules can be initialized in various ways. We demonstrate a few of them
# below.
import torch
from torch import nn
from torch.export import export

import tvm
import tvm_tirx
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

######################################################################
# Import from existing models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The most common way to initialize an IRModule is to import from an existing
# model. Apache TVM accommodates imports from a range of frameworks,
# such as PyTorch and ONNX. This tutorial solely demonstrates the import process
# from PyTorch.


# Create a dummy model
class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


# Give an example argument to torch.export
example_args = (torch.randn(1, 784, dtype=torch.float32),)

# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(TorchModel().eval(), example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
    )

mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
# Print the IRModule
mod_from_torch.show()

######################################################################
# Write with Relax NN Module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apache TVM also provides a set of PyTorch-liked APIs, to help users
# write the IRModule directly.

from tvm.relax.frontend import nn


class RelaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


mod_from_relax, params_from_relax = RelaxModel().export_tvm(
    {"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod_from_relax.show()

######################################################################
# Create via TVMScript
# ~~~~~~~~~~~~~~~~~~~~
# TVMScript is a Python-based DSL for IRModules. We are able to
# directly output the IRModule in the TVMScript syntax, or alternatively,
# parse the TVMScript to obtain an IRModule.

from tvm.relax import script as R
from tvm_tirx.script import ir as I


@I.ir_module
class TVMScriptModule:
    @R.function
    def main(
        x: R.Tensor((1, 784), dtype="float32"),
        fc1_weight: R.Tensor((256, 784), dtype="float32"),
        fc1_bias: R.Tensor((256,), dtype="float32"),
        fc2_weight: R.Tensor((10, 256), dtype="float32"),
        fc2_bias: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims = R.permute_dims(fc1_weight, axes=None)
            matmul = R.matmul(x, permute_dims, out_dtype=None)
            add = R.add(matmul, fc1_bias)
            relu = R.nn.relu(add)
            permute_dims1 = R.permute_dims(fc2_weight, axes=None)
            matmul1 = R.matmul(relu, permute_dims1, out_dtype=None)
            add1 = R.add(matmul1, fc2_bias)
            gv = add1
            R.output(gv)
        return gv


mod_from_script = TVMScriptModule
mod_from_script.show()

######################################################################
# Attributes of an IRModule
# -------------------------
# An IRModule is a collection of functions, indexed by GlobalVars.

mod = mod_from_torch
print(mod.get_global_vars())

######################################################################
# We can access the functions in the IRModule by indexing with the GlobalVars
# or their names

# index by global var name
print(mod["main"])
# index by global var, and checking they are the same function
(gv,) = mod.get_global_vars()
assert mod[gv] == mod["main"]

######################################################################
# Transformations on IRModules
# ----------------------------
# Transformations are the import component of Apache TVM. One transformation
# takes in an IRModule and outputs another IRModule. We can apply a sequence of
# transformations to an IRModule to obtain a new IRModule. That is the common way to
# optimize a model.
#
# In this getting started tutorial, we only demonstrate how to apply transformations
# to an IRModule. For details of each transformation, please refer to the
# :ref:`Transformation API Reference <api-relax-transformation>`

######################################################################
# We first apply **LegalizeOps** transformation to the IRModule. This transformation
# will convert the Relax module into a mixed stage, with both Relax and TensorIR function
# within the same module. Meanwhile, the Relax operators will be converted into ``call_tir``.

mod = mod_from_torch
mod = relax.transform.LegalizeOps()(mod)
mod.show()

######################################################################
# After the transformation, there are much more functions inside the module. Let's print
# the global vars again.

print(mod.get_global_vars())

######################################################################
# Next, Apache TVM provides a set of default transformation pipelines for users,
# to simplify the transformation process. We can then apply the default pipeline to the module.
# The default **zero** pipeline contains very fundamental transformations, including:
#
# - **LegalizeOps**: This transform converts the Relax operators into `call_tir` functions
#   with the corresponding TensorIR Functions. After this transform, the IRModule will
#   contain both Relax functions and TensorIR functions.
# - **AnnotateTIROpPattern**: This transform annotates the pattern of the TensorIR functions,
#   preparing them for subsequent operator fusion.
# - **FoldConstant**: This pass performs constant folding, optimizing operations
#   involving constants.
# - **FuseOps and FuseTIR**: These two passes work together to fuse operators based on the
#   patterns annotated in the previous step (AnnotateTIROpPattern). These passes transform
#   both Relax functions and TensorIR functions.
#
# .. note::
#
#   Here, we have applied **LegalizeOps** twice in the flow. The second time is useless but
#   harmless.
#
#   Every passes can be duplicated in the flow, since we ensure the passes can handle all legal
#   IRModule inputs. This design can help users to construct their own pipeline.

mod = relax.get_pipeline("zero")(mod)
mod.show()

######################################################################
# Deploy the IRModule Universally
# -------------------------------
# After the optimization, we can compile the model into a TVM runtime module.
# Notably, Apache TVM provides the ability of universal deployment, which means
# we can deploy the same IRModule on different backends, including CPU, GPU, and other emerging
# backends.
#
# Deploy on CPU
# ~~~~~~~~~~~~~
# We can deploy the IRModule on CPU by specifying the target as ``llvm``.

exec = tvm.compile(mod, target="llvm")
dev = tvm_tirx.cpu()
vm = relax.VirtualMachine(exec, dev)

raw_data = np.random.rand(1, 784).astype("float32")
data = tvm_tirx.runtime.tensor(raw_data, dev)
cpu_out = vm["main"](data, *params_from_torch["main"]).numpy()
print(cpu_out)

######################################################################
# Deploy on GPU
# ~~~~~~~~~~~~~
# Besides, CPU backend, we can also deploy the IRModule on GPU. GPU requires
# programs containing extra information, such as the thread bindings and shared memory
# allocations. We need a further transformation to generate the GPU programs.
#
# We use ``DLight`` to generate the GPU programs. In this tutorial, we won't go into
# the details of ``DLight``.
#

from tvm.s_tir import dlight as dl

gpu_mod = dl.ApplyDefaultSchedule(
    dl.gpu.Matmul(), dl.gpu.Fallback(), target=tvm_tirx.target.Target("cuda")
)(mod)

######################################################################
# Now we can compile the IRModule on GPU, the similar way as we did on CPU.

exec = tvm.compile(gpu_mod, target="cuda")
dev = tvm_tirx.cuda(0)
vm = relax.VirtualMachine(exec, dev)
# Need to allocate data and params on GPU device
data = tvm_tirx.runtime.tensor(raw_data, dev)
gpu_params = [tvm_tirx.runtime.tensor(p, dev) for p in params_from_torch["main"]]
gpu_out = vm["main"](data, *gpu_params).numpy()
print(gpu_out)

# Check the correctness of the results
assert np.allclose(cpu_out, gpu_out, atol=1e-3)

######################################################################
# Deploy on Other Backends
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Apache TVM also supports other backends, such as different kinds of GPUs
# (Metal, ROCm, Vulkan and OpenCL), different kinds of CPUs (x86, ARM), and other
# emerging backends (e.g., WebAssembly). The deployment process is similar to the
# GPU backend.
