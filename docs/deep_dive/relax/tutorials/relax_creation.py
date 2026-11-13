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
.. _relax-creation:

Relax Creation
==============
This tutorial demonstrates how to create Relax functions and programs.
We'll cover various ways to define Relax functions, including using TVMScript,
and relax NNModule API.
"""

######################################################################
# Create Relax programs using TVMScript
# -------------------------------------
# TVMScript is a domain-specific language for representing Apache TVM's
# intermediate representation (IR). It is a Python dialect that can be used
# to define an IRModule, which contains both TensorIR and Relax functions.
#
# In this section, we will show how to define a simple MLP model with only
# high-level Relax operators using TVMScript.

from tvm import relax, topi
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

n = T.dynamic("n", "int64")


@I.ir_module
class RelaxModule:
    @R.function
    def forward(
        data: R.Tensor((n, 784), dtype="float32"),
        w0: R.Tensor((128, 784), dtype="float32"),
        b0: R.Tensor((128,), dtype="float32"),
        w1: R.Tensor((10, 128), dtype="float32"),
        b1: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor((n, 10), dtype="float32"):
        with R.dataflow():
            lv0 = R.matmul(data, R.permute_dims(w0)) + b0
            lv1 = R.nn.relu(lv0)
            lv2 = R.matmul(lv1, R.permute_dims(w1)) + b1
            R.output(lv2)
        return lv2


RelaxModule.show()

######################################################################
# Relax is not only a graph-level IR, but also supports cross-level
# representation and transformation. To be specific, we can directly call
# TensorIR functions in Relax function.

n = T.dynamic("n", "int64")
m = T.dynamic("m", "int64")


@I.ir_module
class RelaxModuleWithTIR:
    @Ts.function
    def relu(X: T.Tensor((n, m), "float32"), Y: T.Tensor((n, m), "float32")):
        for i, j in T.grid(n, m):
            with Ts.sblock("relu"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))

    @R.function
    def forward(
        data: R.Tensor((n, 784), dtype="float32"),
        w0: R.Tensor((128, 784), dtype="float32"),
        b0: R.Tensor((128,), dtype="float32"),
        w1: R.Tensor((10, 128), dtype="float32"),
        b1: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor((n, 10), dtype="float32"):
        cls = RelaxModuleWithTIR
        with R.dataflow():
            lv0 = R.matmul(data, R.permute_dims(w0)) + b0
            lv1 = R.call_tir(cls.relu, lv0, ty_args=[R.Tensor((n, 128), dtype="float32")])
            lv2 = R.matmul(lv1, R.permute_dims(w1)) + b1
            R.output(lv2)
        return lv2


RelaxModuleWithTIR.show()

######################################################################
# .. note::
#
#   You may notice that the printed output is different from the written
#   TVMScript code. This is because we print the IRModule in a standard
#   format, while we support syntax sugar for the input
#
#   For example, we can combine multiple operators into a single line, as
#
#   .. code-block:: python
#
#     lv0 = R.matmul(data, R.permute_dims(w0)) + b0
#
#   However, the normalized expression requires only one operation in one
#   binding. So the printed output is different from the written TVMScript code,
#   as
#
#   .. code-block:: python
#
#     lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(w0, axes=None)
#     lv1: R.Tensor((n, 128), dtype="float32") = R.matmul(data, lv)
#     lv0: R.Tensor((n, 128), dtype="float32") = R.add(lv1, b0)
#

######################################################################
# Create Relax programs using NNModule API
# ----------------------------------------
# Besides TVMScript, we also provide a PyTorch-like API for defining neural networks.
# It is designed to be more intuitive and easier to use than TVMScript.
#
# In this section, we will show how to define the same MLP model using
# Relax NNModule API.

from tvm.relax.frontend import nn


class NNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


######################################################################
# After we define the NNModule, we can export it to TVM IRModule via
# ``export_tvm``.

mod, params = NNModule().export_tvm({"forward": {"x": nn.spec.Tensor(("n", 784), "float32")}})
mod.show()

######################################################################
# We can also insert customized function calls into the NNModule, such as
# Tensor Expression(TE), TensorIR functions or other TVM packed functions.

M = T.dynamic("M", "int64")
N = T.dynamic("N", "int64")
K = T.dynamic("K", "int64")


@Ts.function
def tir_linear(
    X: T.Tensor((M, K), "float32"),
    W: T.Tensor((N, K), "float32"),
    B: T.Tensor((N,), "float32"),
    Z: T.Tensor((M, N), "float32"),
):
    for i, j, k in T.grid(M, N, K):
        with Ts.sblock("linear"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                Z[vi, vj] = 0
            Z[vi, vj] = Z[vi, vj] + X[vi, vk] * W[vj, vk]
    for i, j in T.grid(M, N):
        with Ts.sblock("add"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Z[vi, vj] = Z[vi, vj] + B[vj]


class NNModuleWithTIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        n = x.shape[0]
        # We can call external functions using nn.extern
        x = nn.extern(
            "env.linear",
            [x, self.fc1.weight, self.fc1.bias],
            out=nn.Tensor.placeholder((n, 128), "float32"),
        )
        # We can also call TensorIR via Tensor Expression API in TOPI
        x = nn.tensor_expr_op(topi.nn.relu, "relu", [x])
        # We can also call other TVM packed functions
        x = nn.tensor_ir_op(
            tir_linear,
            "tir_linear",
            [x, self.fc2.weight, self.fc2.bias],
            out=nn.Tensor.placeholder((n, 10), "float32"),
        )
        return x


mod, params = NNModuleWithTIR().export_tvm(
    {"forward": {"x": nn.spec.Tensor(("n", 784), "float32")}}
)
mod.show()

######################################################################
# Create Relax programs using Block Builder API
# ---------------------------------------------
# In addition to the above APIs, we also provide a Block Builder API for
# creating Relax programs. It is a IR builder API, which is more
# low-level and widely used in TVM's internal logic, e.g writing a
# customized pass.

bb = relax.BlockBuilder()
n = T.dynamic("n", "int64")
x = relax.Var("x", R.Tensor((n, 784), "float32"))
fc1_weight = relax.Var("fc1_weight", R.Tensor((128, 784), "float32"))
fc1_bias = relax.Var("fc1_bias", R.Tensor((128,), "float32"))
fc2_weight = relax.Var("fc2_weight", R.Tensor((10, 128), "float32"))
fc2_bias = relax.Var("fc2_bias", R.Tensor((10,), "float32"))
with bb.function("forward", [x, fc1_weight, fc1_bias, fc2_weight, fc2_bias]):
    with bb.dataflow():
        lv0 = bb.emit(relax.op.matmul(x, relax.op.permute_dims(fc1_weight)) + fc1_bias)
        lv1 = bb.emit(relax.op.nn.relu(lv0))
        gv = bb.emit(relax.op.matmul(lv1, relax.op.permute_dims(fc2_weight)) + fc2_bias)
        bb.emit_output(gv)
    bb.emit_func_output(gv)

mod = bb.get()
mod.show()

######################################################################
# Also, Block Builder API supports building cross-level IRModule with both
# Relax functions, TensorIR functions and other TVM packed functions.

bb = relax.BlockBuilder()
with bb.function("forward", [x, fc1_weight, fc1_bias, fc2_weight, fc2_bias]):
    with bb.dataflow():
        lv0 = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("env.linear"),
                [x, fc1_weight, fc1_bias],
                ty_args=[relax.TensorType((n, 128), "float32")],
            )
        )
        lv1 = bb.emit_te(topi.nn.relu, lv0)
        tir_gv = bb.add_func(tir_linear, "tir_linear")
        gv = bb.emit(
            relax.call_tir(
                tir_gv,
                [lv1, fc2_weight, fc2_bias],
                ty_args=[relax.TensorType((n, 10), "float32")],
            )
        )
        bb.emit_output(gv)
    bb.emit_func_output(gv)
mod = bb.get()
mod.show()

######################################################################
# Note that the Block Builder API is not as user-friendly as the above APIs,
# but it is lowest-level API and works closely with the IR definition. We
# recommend using the above APIs for users who only want to define and
# transform a ML model. But for those who want to build more complex
# transformations, the Block Builder API is a more flexible choice.

######################################################################
# Summary
# -------
# This tutorial demonstrates how to create Relax programs using TVMScript,
# NNModule API, Block Builder API and PackedFunc API for different use cases.
