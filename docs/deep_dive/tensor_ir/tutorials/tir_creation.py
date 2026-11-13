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
.. _tir-creation:

TensorIR Creation
-----------------
In this section, we will introduce the methods to write a TensorIR function
in Apache TVM. This tutorial presumes familiarity with the fundamental concepts of TensorIR.
If not already acquainted, please refer to :ref:`tirx-learning` initially.

.. note::

    This tutorial concentrates on the construction of **standalone** TensorIR functions. The
    techniques presented here are not requisite for end users to compile Relax models.

"""

######################################################################
# Create TensorIR using TVMScript
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The most straightforward way to create a TensorIR function via TVMScript.
# TVMScript is a TVM Python dialect that represents TensorIR in TVM.
#
# .. important::
#
#     While TVMScript employs Python syntax and AST, ensuring full compatibility
#     with Python tools like auto-completion and linting, it is not a native Python
#     language and cannot be executed by a Python interpreter.
#
#     More precisely, the decorator **@tvm.script** extracts the Python AST from
#     the decorated function, subsequently parsing it into TensorIR.
#
# Standard Format
# ***************
# Let's take an example of ``mm_relu`` from :ref:`tirx-learning`. Here is the complete
# format of the ir_module and in TVMScript:

import numpy as np
import tvm_ffi

import tvm
import tvm_tirx
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@I.ir_module
class MyModule:
    @Ts.function
    def mm_relu(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        Y = T.alloc_tensor(ty_args=[T.Tensor((128, 128), "float32")])
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    with Ts.sblock("Y"):
                        vi = Ts.axis.spatial(128, i)
                        vj = Ts.axis.spatial(128, j)
                        vk = Ts.axis.reduce(128, k)
                        Ts.reads(A[vi, vk], B[vk, vj])
                        Ts.writes(Y[vi, vj])
                        with Ts.init():
                            Y[vi, vj] = T.float32(0)
                        Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i in range(128):
            for j in range(128):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(128, i)
                    vj = Ts.axis.spatial(128, j)
                    Ts.reads(Y[vi, vj])
                    Ts.writes(C[vi, vj])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


######################################################################
# Concise with Syntactic Sugar
# ****************************
# For ease of writing, we can employ the following syntactic sugar to
# streamline the code:
#
# - Utilize ``T.grid`` to condense nested loops;
# - Employ ``Ts.axis.remap`` to abbreviate block iterator annotations;
# - Exclude ``Ts.reads`` and ``Ts.writes`` for blocks whose content can
#   be inferred from the block body;


@I.ir_module
class ConciseModule:
    @Ts.function
    def mm_relu(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        Y = T.alloc_tensor(ty_args=[T.Tensor((128, 128), "float32")])
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("Y"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


######################################################################
# We can use the following code to verify that the two modules are equivalent:

print(tvm_ffi.structural_equal(MyModule, ConciseModule))

######################################################################
# Interactive with Python Variables
# *********************************
# Despite TVMScript not being executed by a Python interpreter, limited
# interaction with Python is feasible. For instance, Python variables can
# be used to ascertain the shape and data type of a TensorIR.

# Python variables
M = N = K = 128
dtype = "float32"


# IRModule in TVMScript
@I.ir_module
class ConciseModuleFromPython:
    @Ts.function
    def mm_relu(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        Y = T.alloc_tensor(ty_args=[T.Tensor((M, N), dtype)])
        for i, j, k in T.grid(M, N, K):
            with Ts.sblock("Y"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    Y[vi, vj] = T.cast(T.float32(0), dtype)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(M, N):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.cast(T.float32(0), dtype))


######################################################################
# Check the equivalence:

print(tvm_ffi.structural_equal(ConciseModule, ConciseModuleFromPython))

######################################################################
# TensorIR Function with Dynamic Shapes
# *************************************
# Despite TVMScript not being executed by a Python interpreter, limited
# interaction with Python is feasible. For instance, Python variables can
# be used to ascertain the shape and data type of a TensorIR.

# Dynamic shape definition
M = T.dynamic("M", "int32")
N = T.dynamic("N", "int32")
K = T.dynamic("K", "int32")


@I.ir_module
class DynamicShapeModule:
    @Ts.function
    def mm_relu(A: T.Tensor([M, K], dtype), B: T.Tensor([K, N], dtype), C: T.Tensor([M, N], dtype)):
        # Bind the input buffers with the dynamic shapes

        Y = T.alloc_tensor(ty_args=[T.Tensor((M, N), dtype)])
        for i, j, k in T.grid(M, N, K):
            with Ts.sblock("Y"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    Y[vi, vj] = T.cast(T.float32(0), dtype)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(M, N):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.cast(T.float32(0), dtype))


######################################################################
# Now let's check the runtime dynamic shape inference:


def evaluate_dynamic_shape(lib: tvm_ffi.Module, m: int, n: int, k: int):
    A = tvm_tirx.runtime.tensor(np.random.uniform(size=(m, k)).astype("float32"))
    B = tvm_tirx.runtime.tensor(np.random.uniform(size=(k, n)).astype("float32"))
    C = tvm_tirx.runtime.tensor(np.zeros((m, n), dtype="float32"))
    lib(A, B, C)
    return C.numpy()


# Compile lib only once
dyn_shape_lib = tvm.compile(DynamicShapeModule, target="llvm")
# Able to handle different shapes
print(evaluate_dynamic_shape(dyn_shape_lib, m=4, n=4, k=4))
print(evaluate_dynamic_shape(dyn_shape_lib, m=64, n=64, k=128))

######################################################################
# Create TensorIR using Tensor Expression
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Often, the specifics of TensorIR are disregarded in favor of expressing the computation more
# succinctly, leading to the pragmatic generation of TensorIR. This is where Tensor Expression
# (TE) becomes relevant.
#
# Tensor Expression (TE) serves as a domain-specific language delineating a sequence of
# computations through an expression-like API.
#
# .. note::
#
#   Tensor Expression comprises two components within the TVM stack: the expression and the
#   schedule. The expression is the domain-specific language embodying the computation pattern,
#   precisely what we're addressing in this section. Conversely, the TE schedule is the legacy
#   scheduling method, has been superseded by the TensorIR schedule in the current TVM stack.
#
# Create Static-Shape Functions
# *****************************
# We use the same example of ``mm_relu`` from the last subsection to demonstrate the
# TE creation method.

from tvm import te

A = te.placeholder((128, 128), "float32", name="A")
B = te.placeholder((128, 128), "float32", name="B")
k = te.reduce_axis((0, 128), "k")
Y = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C = te.compute((128, 128), lambda i, j: te.max(Y[i, j], 0), name="C")

######################################################################
# Here ``te.compute`` takes the signature ``te.compute(output_shape, fcompute)``.
# And the fcompute function describes how we want to compute the value of each
# element ``Y[i, j]`` for a given index:
#
# .. code:: python
#
#   lambda i, j: te.sum(A[i, k] * B[k, j], axis=k)
#
# The aforementioned lambda expression encapsulates the computation:
# :math:`Y_{i, j} = \sum_k A_{i, k} \times B_{k, j}`. Upon defining the computation,
# we can formulate a TensorIR function by incorporating the pertinent parameters of interest.
# In this specific instance, we aim to construct a function with two input parameters **A, B**
# and one output parameter **C**.

te_func = te.create_function([A, B, C]).with_attr({"global_symbol": "mm_relu"})
TEModule = tvm_tirx.IRModule({"mm_relu": te_func})
TEModule.show()

######################################################################
# Create Dynamic-Shape Functions
# ******************************
# We can also create a dynamic-shape function using Tensor Expression. The only difference
# is that we need to specify the shape of the input tensors as symbolic variables.

# Declare symbolic variables
M, N, K = te.var("m"), te.var("n"), te.var("k")
A = te.placeholder((M, N), "float32", name="A")
B = te.placeholder((K, N), "float32", name="B")
k = te.reduce_axis((0, K), "k")
Y = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C = te.compute((M, N), lambda i, j: te.max(Y[i, j], 0), name="C")

dyn_te_func = te.create_function([A, B, C]).with_attr({"global_symbol": "mm_relu"})
DynamicTEModule = tvm_tirx.IRModule({"mm_relu": dyn_te_func})
DynamicTEModule.show()
