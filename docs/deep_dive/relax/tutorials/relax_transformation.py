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
.. _relax-transform:

Transformation
--------------
In this section, we will dive into the transformation of Relax programs.
Transformations is one of the key ingredients of the compilation flows
for optimizing and integrating with hardware backends.
"""

######################################################################
# Let's first create a simple Relax program as what we have done in
# the :ref:`previous section <relax-creation>`.

import tvm
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm_tirx import IRModule


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


origin_mod, params = NNModule().export_tvm(
    {"forward": {"x": nn.spec.Tensor(("n", 784), "float32")}}
)
origin_mod.show()

######################################################################
# Apply transformations
# ~~~~~~~~~~~~~~~~~~~~~
# Passes are the main way to apply transformations to the program.
# We can apply passes to the program. As first step, let's apply
# a built-in pass ``LegalizeOps`` to lower the high-level operators
# into low-level operators.

mod = tvm.relax.transform.LegalizeOps()(origin_mod)
mod.show()

######################################################################
# As we can see from the output, the high-level operators (aka ``relax.op``) in the program
# are replaced by their corresponding low-level operators (aka ``relax.call_tir``).
#
# Then let's trying to apply the operator fusion, which is a wide-used optimization technique
# in ML compilers. Note that in relax, fusion optimizations are done with the collaboration of
# a set of passes. We can apply them in a sequence.

mod = tvm_tirx.ir.transform.Sequential(
    [
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),
    ]
)(mod)
mod.show()

######################################################################
# As result, we can see that the ``matmul``, ``add`` and ``relu`` operators are fused
# into one kernel (aka one ``call_tir``).
#
# For all built-in passes, please refer to :py:class:`relax.transform`.
#
# Custom Passes
# ~~~~~~~~~~~~~
# We can also define our own passes. Let's take an example of rewriting the ``relu``
# operator to ``gelu`` operator.
#
# First, we need to write a Relax IR Mutator to do the rewriting.

from tvm.relax.expr_functor import PyExprMutator, mutator


@mutator
class ReluRewriter(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        # visit the relax.Call expr, and only handle the case when op is relax.nn.relu
        if call.op.name == "relax.nn.relu":
            return relax.op.nn.gelu(call.args[0])

        return super().visit_call_(call)


######################################################################
# Then we can write a pass to apply the mutator to the whole module.


@tvm_tirx.ir.transform.module_pass(opt_level=0, name="ReluToGelu")
class ReluToGelu:  # pylint: disable=too-few-public-methods
    def transform_module(self, mod: IRModule, _ctx: tvm_tirx.ir.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        rewriter = ReluRewriter(mod)
        for g_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = rewriter.visit_expr(func)
                rewriter.builder_.update_func(g_var, func)
        return rewriter.builder_.get()


mod = ReluToGelu()(origin_mod)
mod.show()

######################################################################
# The printed output shows that the ``relax.nn.relu`` operator is
# rewritten to ``relax.nn.gelu`` operator.
#
# For the details of the mutator, please refer to :py:class:`relax.expr_functor.PyExprMutator`.
#
# Summary
# ~~~~~~~
# In this section, we have shown how to apply transformations to the Relax program.
# We have also shown how to define and apply custom transformations.
