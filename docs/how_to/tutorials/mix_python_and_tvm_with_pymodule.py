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
.. _mix_python_and_tvm:

Mix Python/PyTorch with TVM Using BasePyModule
===============================================
In a typical TVM workflow, you write an ``IRModule``, compile it, and load the compiled artifact
into a ``VirtualMachine`` to run. This means **you cannot test or debug anything until the entire
module compiles successfully**. If a single op is unsupported, the whole pipeline is blocked.

``BasePyModule`` solves this by letting Python functions, TIR kernels, and Relax functions coexist
in one module. TIR and Relax functions are JIT-compiled on instantiation, Python functions run
as-is, and tensors move between TVM and PyTorch via zero-copy DLPack. This enables:

- **Incremental development**: get a model running with Python fallbacks first, then replace them
  with TVM ops one by one.
- **Easy debugging**: insert ``print`` in Python functions to inspect intermediate tensors — no
  need to compile the whole module first.
- **Verification at any compilation stage**: convert Relax IR back to PyTorch to check numerical
  correctness before and after optimization passes.
- **Hybrid execution**: let the compiled VM call back into Python for ops that are hard to
  express in TIR or Relax.

This tutorial walks through the full workflow step by step.

.. contents:: Table of Contents
    :local:
    :depth: 1
"""

######################################################################
# Preparation
# -----------
import os

import tvm_ffi

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

import tvm_tirx
from tvm import relax
from tvm.relax import script as R
from tvm.relax.base_py_module import BasePyModule
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

IS_IN_CI = os.getenv("CI", "").lower() == "true"
HAS_TORCH = torch is not None
RUN_EXAMPLE = HAS_TORCH and not IS_IN_CI

######################################################################
# Step 1: Your First Hybrid Module
# ----------------------------------
# The core idea: decorate a class with ``@R.py_module``, inherit from ``BasePyModule``, and use
# three decorators for three kinds of functions:
#
# - ``@T.function`` — low-level TIR kernel (JIT-compiled on instantiation)
# - ``@R.function`` — high-level Relax graph (JIT-compiled on instantiation)
# - ``@I.pyfunc`` — plain Python (runs as-is, can use any Python library)
#
# ``call_tir`` bridges Python and TIR: it converts PyTorch tensors to TVM NDArrays via DLPack
# (zero-copy), allocates the output buffer, calls the compiled kernel, and converts back.

if RUN_EXAMPLE:

    @R.py_module
    class MyFirstModule(BasePyModule):
        @Ts.function
        def add_tir(
            A: T.Tensor((4,), "float32"),
            B: T.Tensor((4,), "float32"),
            C: T.Tensor((4,), "float32"),
        ):
            for i in range(4):
                C[i] = A[i] + B[i]

        @I.pyfunc
        def forward(self, x, y):
            """Takes PyTorch tensors, calls TIR, returns PyTorch tensors."""
            x_tvm = self._convert_pytorch_to_tvm(x)
            y_tvm = self._convert_pytorch_to_tvm(y)
            result = self.call_tir(self.add_tir, [x_tvm, y_tvm], out_ty=R.Tensor((4,), "float32"))
            return self._convert_tvm_to_pytorch(result)

    # TIR functions are JIT-compiled at instantiation
    mod = MyFirstModule(device=tvm_tirx.cpu(0))

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0])
    result = mod.forward(x, y)

    print("forward(x, y) =", result)
    assert torch.allclose(result, x + y)

    # show() prints TVMScript including Python functions (shown as ExternFunc)
    mod.show()

    # list_functions() shows what is available in the module
    print("Available functions:", mod.list_functions())

######################################################################
# Step 2: Debugging — The Main Selling Point
# ---------------------------------------------
# Traditional ML compilers treat computation graphs as monolithic blobs. You cannot inspect
# intermediate tensor values without compiling the entire module. With ``@I.pyfunc``, debugging
# is as simple as adding a ``print`` statement. You can also make quick edits and re-run
# immediately — no recompilation needed.

if RUN_EXAMPLE:
    n = T.dynamic("n", "int32")

    @R.py_module
    class DebugModule(BasePyModule):
        @Ts.function
        def matmul_tir(
            A: T.Tensor((n, 4), "float32"),
            B: T.Tensor((4, 3), "float32"),
            C: T.Tensor((n, 3), "float32"),
        ):
            for i, j, k in T.grid(n, 3, 4):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @I.pyfunc
        def forward(self, x, weights):
            # Inspect input
            print(f"  [DEBUG] input shape: {x.shape}, mean: {x.mean():.4f}")

            # Run TIR matmul
            x_tvm = self._convert_pytorch_to_tvm(x)
            w_tvm = self._convert_pytorch_to_tvm(weights)
            out = self.call_tir(
                self.matmul_tir,
                [x_tvm, w_tvm],
                out_ty=R.Tensor((x.shape[0], 3), "float32"),
            )
            logits = self._convert_tvm_to_pytorch(out)

            # Inspect intermediate value — impossible with a compiled-only workflow
            print(
                f"  [DEBUG] logits shape: {logits.shape}, "
                f"min: {logits.min():.4f}, max: {logits.max():.4f}"
            )

            result = F.softmax(logits, dim=-1)

            # Verify output
            print(f"  [DEBUG] probs sum: {result.sum(dim=-1)}")
            return result

    mod = DebugModule(device=tvm_tirx.cpu(0))

    print("Running with debug prints:")
    probs = mod.forward(torch.randn(2, 4), torch.randn(4, 3))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

######################################################################
# This is the key benefit: "debugging is as simple as inserting a print statement.
# Users can also make quick, manual edits to Python functions and immediately observe the
# results." No compilation cycle, no VM loading — just Python.

######################################################################
# Step 3: A Realistic Pipeline — Python, TIR, and Packed Functions
# -------------------------------------------------------------------
# Real models combine many kinds of operations. This step builds a mini inference pipeline using
# three different calling conventions:
#
# - ``call_tir``: call a compiled TIR kernel
# - ``call_dps_packed``: call a TVM packed function (e.g., a third-party library binding)
# - Direct Python: call any PyTorch function
#
# ``call_dps_packed`` can call functions registered via
# ``tvm_ffi.register_global_func``
# — for example, CUBLAS or cuDNN bindings that TVM wraps as packed functions.

if RUN_EXAMPLE:
    # Register a packed function (simulating an external library binding)
    @tvm_ffi.register_global_func("my_bias_add", override=True)
    def my_bias_add(x, bias, out):
        """Packed function: adds bias to each row of x."""

        x_np = x.numpy()
        b_np = bias.numpy()
        out_np = x_np + b_np
        out[:] = out_np

    @R.py_module
    class PipelineModule(BasePyModule):
        @Ts.function
        def matmul_tir(
            A: T.Tensor((2, 4), "float32"),
            B: T.Tensor((4, 3), "float32"),
            C: T.Tensor((2, 3), "float32"),
        ):
            for i, j, k in T.grid(2, 3, 4):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @I.pyfunc
        def forward(self, x, weights, bias):
            # 1. TIR matmul
            x_tvm = self._convert_pytorch_to_tvm(x)
            w_tvm = self._convert_pytorch_to_tvm(weights)
            h = self.call_tir(
                self.matmul_tir,
                [x_tvm, w_tvm],
                out_ty=R.Tensor((2, 3), "float32"),
            )
            h_pt = self._convert_tvm_to_pytorch(h)

            # 2. Packed function for bias add (simulating an external library)
            h_biased = self.call_dps_packed(
                "my_bias_add",
                [h_pt, bias],
                out_ty=R.Tensor((2, 3), "float32"),
            )

            # 3. Python/PyTorch activation
            return F.relu(h_biased)

    mod = PipelineModule(device=tvm_tirx.cpu(0))

    x = torch.randn(2, 4)
    w = torch.randn(4, 3)
    b = torch.randn(3)
    result = mod.forward(x, w, b)

    expected = F.relu(x @ w + b)
    print("Pipeline result:", result)
    print("Expected:       ", expected)
    assert torch.allclose(result, expected, atol=1e-4)

######################################################################
# Step 4: Relax-to-Python Converter — Verify at Any Compilation Stage
# ----------------------------------------------------------------------
# Both Relax functions and Python functions describe computational graphs. The
# ``RelaxToPyFuncConverter`` converts Relax IR into equivalent PyTorch code by mapping
# Relax operators to their PyTorch counterparts (e.g., ``R.nn.relu`` → ``F.relu``).
#
# A key feature: **this conversion can happen at any stage of compilation**.
# You can convert early (right after import) or late (after optimization passes have
# transformed the IR), and compare the output against a PyTorch reference to catch bugs.

if RUN_EXAMPLE:
    from tvm.relax.relax_to_pyfunc_converter import RelaxToPyFuncConverter

    # A simple Relax module: matmul + bias + relu (a dense layer)
    @I.ir_module
    class DenseLayer:
        @Ts.function
        def bias_add_tir(
            x: T.Tensor((2, 4), "float32"),
            b: T.Tensor((4,), "float32"),
            out: T.Tensor((2, 4), "float32"),
        ):
            for i, j in T.grid(2, 4):
                out[i, j] = x[i, j] + b[j]

        @R.function
        def main(
            x: R.Tensor((2, 4), "float32"),
            w: R.Tensor((4, 4), "float32"),
            b: R.Tensor((4,), "float32"),
        ) -> R.Tensor((2, 4), "float32"):
            h = R.matmul(x, w)
            cls = DenseLayer
            h_bias = R.call_tir(
                cls.bias_add_tir,
                (h, b),
                ty_args=[R.Tensor((2, 4), "float32")],
            )
            return R.nn.relu(h_bias)

    # --- Stage 1: Convert BEFORE optimization ---
    converter = RelaxToPyFuncConverter(DenseLayer)
    converted_early = converter.convert(["main"])

    x = torch.randn(2, 4)
    w = torch.randn(4, 4)
    b = torch.randn(4)

    py_result_early = converted_early.__pyfuncs__["main"](x, w, b)
    expected = F.relu(x @ w + b)

    print("Before optimization:")
    print("  Converted result:", py_result_early)
    print("  PyTorch expected:", expected)
    assert torch.allclose(py_result_early, expected, atol=1e-5)

    # --- Stage 2: Apply a pass, then convert AFTER optimization ---
    # Run CanonicalizeBindings to clean up the IR, then convert again
    # to verify the pass did not break numerical correctness.
    optimized_mod = relax.transform.CanonicalizeBindings()(DenseLayer)

    converter_late = RelaxToPyFuncConverter(optimized_mod)
    converted_late = converter_late.convert(["main"])

    py_result_late = converted_late.__pyfuncs__["main"](x, w, b)

    print("\nAfter CanonicalizeBindings pass:")
    print("  Converted result:", py_result_late)
    print("  Still matches:   ", torch.allclose(py_result_late, expected, atol=1e-5))
    assert torch.allclose(py_result_late, expected, atol=1e-5)

######################################################################
# Step 5: R.call_py_func — Python Callbacks in Compiled IR
# -----------------------------------------------------------
# ``R.call_py_func`` embeds a Python function call directly inside Relax IR. When the module
# is compiled and run in the VM, everything else is optimized native code, but the VM calls
# back into Python for the specified ops.
#
# ``BasePyModule`` supports cross-level calls in both directions: Relax functions can invoke
# Python functions, and Python functions can invoke TIR/Relax functions. Data flows between
# them via DLPack with minimal overhead.
#
# Use case: your model has a custom op (e.g., a special normalization or a sampling step)
# that is complex to implement in TIR. Compile everything else, and let that one op stay
# in Python.

if RUN_EXAMPLE:

    @R.py_module
    class HybridVMModule(BasePyModule):
        @I.pyfunc
        def silu(self, x):
            """SiLU/Swish activation — using Python as fallback."""
            return torch.sigmoid(x) * x

        @I.pyfunc
        def layer_norm(self, x):
            """LayerNorm — another Python fallback."""
            return F.layer_norm(x, x.shape[-1:])

        @R.function
        def main(
            x: R.Tensor((4, 8), "float32"),
        ) -> R.Tensor((4, 8), "float32"):
            # The VM calls back into Python for these two ops
            h = R.call_py_func("layer_norm", (x,), ty_args=[R.Tensor((4, 8), "float32")])
            out = R.call_py_func("silu", (h,), ty_args=[R.Tensor((4, 8), "float32")])
            return out

    mod = HybridVMModule(device=tvm_tirx.cpu(0))
    x = torch.randn(4, 8)

    # call_py_func is also callable from Python directly
    result = mod.call_py_func("layer_norm", [x])
    result = mod.call_py_func("silu", [result])

    ln = F.layer_norm(x, x.shape[-1:])
    expected = torch.sigmoid(ln) * ln
    print("call_py_func result:", result)
    assert torch.allclose(torch.tensor(result.numpy()), expected, atol=1e-5)

######################################################################
# Step 6: Cross-Level Calls and Symbolic Shapes
# ------------------------------------------------
# ``BasePyModule`` is designed for **cross-level interoperability**: Python functions can call
# TIR and Relax functions, and Relax functions can call Python functions. We have already seen:
#
# - Python → TIR via ``call_tir`` (Steps 1-3)
# - Python → packed function via ``call_dps_packed`` (Step 3)
# - Relax → Python via ``R.call_py_func`` (Step 5)
#
# The missing piece: **Python calling a compiled Relax function directly**. When a module
# contains ``@R.function``, it is JIT-compiled into a Relax VM. You can call it from Python
# just like any other method — the module auto-converts PyTorch tensors to TVM and back.
#
# This step also shows **symbolic shapes**: TIR and Relax functions can declare dynamic
# dimensions (e.g., ``"n"``). ``BasePyModule`` infers concrete shapes from the actual input
# tensors at call time, so the same module handles different sizes without recompilation.

if RUN_EXAMPLE:
    n = T.dynamic("n", "int64")

    @R.py_module
    class DynamicModule(BasePyModule):
        @Ts.function
        def scale_tir(x: T.Tensor((n,), "float32"), out: T.Tensor((n,), "float32")):
            for i in T.serial(n):
                out[i] = x[i] * T.float32(2.0)

        @R.function
        def add_relax(
            x: R.Tensor((n,), "float32"),
            y: R.Tensor((n,), "float32"),
        ) -> R.Tensor((n,), "float32"):
            return R.add(x, y)

    mod = DynamicModule(device=tvm_tirx.cpu(0), target="llvm")

    # Inspect what the module contains
    print("Functions:", mod.list_functions())

    # Python → Relax: call the compiled Relax function directly with PyTorch tensors
    a5 = torch.randn(5)
    b5 = torch.randn(5)
    out5 = mod.add_relax(a5, b5)
    print("add_relax(len=5):", out5)

    # Same module, different size — symbolic shapes handle this automatically
    a10 = torch.randn(10)
    b10 = torch.randn(10)
    out10 = mod.add_relax(a10, b10)
    print("add_relax(len=10):", out10)

    # Python → TIR with symbolic output shape
    n = T.dynamic("n", "int64")
    x7 = torch.randn(7)
    scaled = mod.call_tir("scale_tir", [x7], relax.TensorType((n,), "float32"))
    print("scale_tir(len=7):", scaled)
    assert torch.allclose(torch.tensor(scaled.numpy()), x7 * 2.0, atol=1e-5)

######################################################################
# Summary
# -------
# Cross-level call summary:
#
# - **Python → TIR**: ``call_tir()`` (Steps 1, 2, 3, 6)
# - **Python → packed function**: ``call_dps_packed()`` (Step 3)
# - **Python → Relax**: call ``@R.function`` as a method (Step 6)
# - **Relax → Python**: ``R.call_py_func()`` in compiled VM (Step 5)
#
# The workflow in practice:
#
# 1. Import a model → some ops unsupported → use ``@I.pyfunc`` as Python fallbacks
# 2. Get it running end-to-end with ``BasePyModule``
# 3. Debug by inserting ``print`` in pyfuncs — inspect intermediate tensors instantly
# 4. Use ``RelaxToPyFuncConverter`` to verify correctness after each optimization pass
# 5. Gradually replace Python fallbacks with TIR/Relax implementations
# 6. Use ``R.call_py_func`` for ops that must stay in Python even after compilation
