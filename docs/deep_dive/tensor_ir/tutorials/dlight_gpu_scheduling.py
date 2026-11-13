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
# ruff: noqa: E402, E501

"""
.. _dlight_gpu_scheduling:

DLight: Rule-Based GPU Scheduling
==================================
TIR functions produced by Relax legalization need GPU-specific scheduling — thread binding,
loop tiling, shared memory usage — before they can run efficiently on a GPU. There are two
main approaches in TVM:

- **MetaSchedule**: explores a search space to find the best schedule. High quality, but
  compilation takes minutes to hours.
- **DLight**: applies pre-defined scheduling rules deterministically. No tuning required,
  compilation completes in seconds. Performance is excellent for well-known patterns
  (e.g., GEMM, GEMV in LLM workloads) and fair for the rest.

This tutorial covers how DLight works, what rules are available, how to diagnose scheduling
quality, and how to write custom rules.

.. contents:: Table of Contents
    :local:
    :depth: 1
"""

######################################################################
# Prepare a Model
# ---------------
# We build a small model with ``nn.Module`` that is rich enough to trigger multiple DLight
# rules: ``Linear`` layers produce GEMM (matrix multiplication) kernels, ``LayerNorm``
# produces a general-reduction kernel, and ``ReLU`` is a simple elementwise op.

import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm.s_tir import dlight as dl
from tvm_tirx import tirx


class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(768)
        self.fc2 = nn.Linear(768, 256)

    def forward(self, x):
        x = self.norm(self.relu(self.fc1(x)))
        return self.fc2(x)


mod, params = DemoModel().export_tvm({"forward": {"x": nn.spec.Tensor((1, 768), "float32")}})

######################################################################
# Legalize Relax operators into TIR functions so that DLight has concrete kernels to schedule.

device = tvm_tirx.cuda(0)
target = tvm_tirx.target.Target.from_device(device)
mod = relax.get_pipeline("zero", target=target)(mod)

######################################################################
# At this point every TIR function in ``mod`` is **unscheduled** — it has no thread bindings
# and would not run efficiently on a GPU. Let's see what functions we have:
for gv, func in mod.functions_items():
    if isinstance(func, tirx.Function):
        print(f"  {gv.name_hint}")

######################################################################
# Basic Usage: ApplyDefaultSchedule
# ---------------------------------
# ``ApplyDefaultSchedule`` is an ``IRModule`` pass. It iterates over every TIR function in the
# module and tries the given rules **in order**. For each function the first rule whose
# ``apply()`` returns a non-``None`` schedule wins; subsequent rules are skipped.
# After scheduling, the function is marked with ``tirx.is_scheduled`` so it won't be
# scheduled again by a later ``ApplyDefaultSchedule`` call.

######################################################################
# Here we use a common subset of rules. The full catalog (including ``LowBatchGEMV``,
# ``Transpose``, ``RMSNorm``) is listed in the next section.

scheduled_mod = dl.ApplyDefaultSchedule(
    dl.gpu.Matmul(),  # GEMM: dense matrix multiplication
    dl.gpu.GEMV(),  # matrix-vector products
    dl.gpu.Reduction(),  # simple reductions (sum, max, ...)
    dl.gpu.GeneralReduction(),  # compound reductions (softmax, layer norm, ...)
    dl.gpu.Fallback(),  # catch-all for anything unmatched above
    target=target,
)(mod)

scheduled_mod.show()

######################################################################
# Compared with the unscheduled IR, you can now see thread bindings
# (``blockIdx.x``, ``threadIdx.x``, ...) and loop transformations in each TIR function.

######################################################################
# Rule Catalog
# ------------
# DLight ships a set of GPU scheduling rules. Each rule is a subclass of
# ``ScheduleRule`` and implements an ``apply(func, target, tunable)`` method that returns
# a ``Schedule`` if the rule matches, or ``None`` to pass.
#
# The built-in GPU rules, roughly from most specific to most general:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 20 40 40
#
#    * - Rule
#      - Pattern
#      - Typical operators
#    * - ``Matmul``
#      - GEMM index pattern ``C[S,I,J] += A[S,I,K] * B[S,J,K]``
#      - ``nn.Linear``, batched matmul
#    * - ``GEMV``
#      - Matrix-vector multiply (one dimension is 1)
#      - single-batch decode in attention
#    * - ``LowBatchGEMV``
#      - Low-batch GEMM scheduled with a GEMV strategy
#      - small-batch decode
#    * - ``Reduction``
#      - Simple accumulation ``X[...] += Y[...]``
#      - sum, max, argmax
#    * - ``GeneralReduction``
#      - Spatial dims followed by reduction dims (``S* R*``)
#      - softmax, layer norm, RMS norm
#    * - ``Transpose``
#      - Read/write indices are permutations of each other
#      - 2-D transpose
#    * - ``RMSNorm``
#      - Contains an ``rsqrt`` operation
#      - RMS normalization
#    * - ``Fallback``
#      - Any function (always matches)
#      - generic catch-all
#
# **Rule order matters.** ``ApplyDefaultSchedule`` stops at the first match, so:
#
# - Put **specialized** rules first (``Matmul``, ``GEMV``) — they have strict matching
#   conditions but produce high-quality schedules.
# - Put **general** rules later (``GeneralReduction``, ``Fallback``) — they match broadly
#   but with less optimal schedules.
# - If you put ``Fallback`` first, it would "steal" every function and no specialized
#   rule would ever run.

######################################################################
# Diagnosing Schedule Quality
# ---------------------------
# A common question is: *which rule scheduled which function?* ``ApplyDefaultSchedule``
# does not log this directly, but you can figure it out by applying rules one at a time.
#
# **Step 1**: Apply each rule individually and record which functions it claims.

from collections import OrderedDict

rules = OrderedDict(
    [
        ("Matmul", dl.gpu.Matmul()),
        ("GEMV", dl.gpu.GEMV()),
        ("LowBatchGEMV", dl.gpu.LowBatchGEMV()),
        ("Reduction", dl.gpu.Reduction()),
        ("GeneralReduction", dl.gpu.GeneralReduction()),
        ("Transpose", dl.gpu.Transpose()),
        ("RMSNorm", dl.gpu.RMSNorm()),
    ]
)

rule_assignment = {}
for rule_name, rule in rules.items():
    test_mod = dl.ApplyDefaultSchedule(rule, target=target)(mod)
    for gv, func in test_mod.functions_items():
        if isinstance(func, tirx.Function) and gv.name_hint not in rule_assignment:
            if "tirx.is_scheduled" in func.attrs and func.attrs["tirx.is_scheduled"] == 1:
                rule_assignment[gv.name_hint] = rule_name

######################################################################
# **Step 2**: Functions not claimed by any specialized rule will fall through to ``Fallback``.

all_tir_funcs = [
    gv.name_hint for gv, func in mod.functions_items() if isinstance(func, tirx.Function)
]
fallback_funcs = [name for name in all_tir_funcs if name not in rule_assignment]

print("Rule assignments:")
for name, rule_name in sorted(rule_assignment.items()):
    print(f"  {name:40s} -> {rule_name}")
if fallback_funcs:
    print("Handled by Fallback (may have suboptimal performance):")
    for name in sorted(fallback_funcs):
        print(f"  {name}")

######################################################################
# If an important kernel lands in the Fallback bucket, you have three options:
#
# 1. Write a **custom DLight rule** for it (see below).
# 2. Use **MetaSchedule** to auto-tune that specific function.
# 3. Manually schedule it with the ``tvm.s_tir.Schedule`` API.

######################################################################
# DLight vs MetaSchedule
# ----------------------
# The two systems are complementary, not competing:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 20 40 40
#
#    * -
#      - DLight
#      - MetaSchedule
#    * - Mechanism
#      - Deterministic rule matching
#      - Search-space exploration
#    * - Compile time
#      - Seconds
#      - Minutes to hours
#    * - Performance
#      - Excellent on known patterns, fair otherwise
#      - Near-optimal with sufficient search budget
#    * - Best for
#      - Default path, rapid iteration, CI
#      - Hot-spot tuning in production
#
# A practical workflow:
#
# 1. Run ``ApplyDefaultSchedule`` with the full rule set to cover all functions.
# 2. Profile the compiled model to identify hot-spot kernels.
# 3. Use ``MetaScheduleTuneTIR`` to auto-tune only those kernels.
#
# Note that ``MetaScheduleTuneTIR`` does **not** automatically skip functions already
# scheduled by DLight — it processes every ``Function`` in the module. In practice this
# is harmless (tuning an already-scheduled function simply re-explores its space), but if
# you want to avoid the extra search cost, filter the module or use ``MetaScheduleTuneIRMod``
# with ``op_names`` to target specific functions.

######################################################################
# Writing a Custom Rule
# ---------------------
# You can extend DLight by writing your own ``ScheduleRule``. The simplest way is
# ``ScheduleRule.from_callable``, which wraps a plain function into a rule **instance**.

from tvm import s_tir
from tvm.s_tir.dlight.analysis import normalize_function
from tvm.s_tir.dlight.base.schedule_rule import ScheduleRule


@ScheduleRule.from_callable("MyTileAndBind")
def my_tile_and_bind(func: tirx.Function, target: tvm_tirx.target.Target, tunable: bool):
    """A minimal rule: for single-block injective functions, tile and bind to GPU threads."""
    if not isinstance(func, tirx.Function):
        return None
    sch = s_tir.Schedule(func)
    # Use normalize_function to get block info with correct spatial/reduction classification.
    # This is the same analysis used by built-in DLight rules.
    block_infos = normalize_function(sch)
    if block_infos is None or len(block_infos) != 1:
        return None  # only handle single-block functions
    info = block_infos[0]
    if not info.is_injective():
        return None  # skip reductions — dom_kind() uses iter_type, not loop kind
    loops = sch.get_loops(info.block_rv)
    if len(loops) == 0:
        return None
    fused = sch.fuse(*loops)
    bx, tx = sch.split(fused, factors=[None, 256])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    return sch


######################################################################
# Insert the custom rule into the rule chain. Note that ``from_callable`` returns an
# **instance**, so pass it directly — do not call ``my_tile_and_bind()`` again.

custom_mod = dl.ApplyDefaultSchedule(
    dl.gpu.Matmul(),
    dl.gpu.GeneralReduction(),
    my_tile_and_bind,  # our custom rule, tried before Fallback
    dl.gpu.Fallback(),
    target=target,
)(mod)

custom_mod.show()

######################################################################
# To build a production-quality rule, subclass ``ScheduleRule`` directly and implement
# ``apply()`` with full analysis logic (see ``tvm.s_tir.dlight.gpu.Matmul`` for an example).

######################################################################
# Summary
# -------
# - **DLight** provides fast, deterministic GPU scheduling via rule matching.
# - Rules are tried in order; the first match wins. Put specialized rules before general ones.
# - Use the **single-rule probing** technique to diagnose which rule handles each function.
# - Combine DLight with MetaSchedule: DLight for baseline coverage, MetaSchedule for hot-spot tuning.
# - Extend DLight by writing custom ``ScheduleRule`` implementations.
#
# For DLight's role in the broader optimization pipeline, see :ref:`customize_opt`.
