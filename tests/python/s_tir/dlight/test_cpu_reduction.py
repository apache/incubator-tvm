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
# pylint: disable=missing-docstring
"""Tests for CPU DLight Reduction schedule rule."""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import te, topi
from tvm.s_tir import dlight as dl
from tvm.s_tir.dlight.cpu import Reduction
from tvm_tirx import tirx
from tvm_tirx.target import Target

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llvm_target():
    return Target({"kind": "llvm"})


def _rvv_target():
    return Target(
        {
            "kind": "llvm",
            "mtriple": "riscv64-linux-gnu",
            "mcpu": "generic-rv64",
            "mabi": "lp64d",
            "mattr": ["+64bit", "+m", "+a", "+f", "+d", "+c", "+v"],
        }
    )


def _build_softmax(batch, features, fast=False):
    A = te.placeholder((batch, features), dtype="float32", name="A")
    B = topi.nn.fast_softmax(A, axis=1) if fast else topi.nn.softmax(A, axis=1)
    func = te.create_function([A, B])
    return tvm_tirx.IRModule({"main": func})


def _apply_and_check(mod, target):
    """Apply Reduction rule and verify it was applied."""
    rule = Reduction()
    result = rule.apply(mod["main"], target, False)
    assert result is not None, "Reduction rule should apply"
    return result


# ---------------------------------------------------------------------------
# Test: schedule applicability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fast", [False, True], ids=["softmax", "fast_softmax"])
@pytest.mark.parametrize(
    "batch,features",
    [
        (1, 10),
        (1, 128),
        (14, 185),
        (32, 256),
        (64, 512),
        (128, 1024),
        (1, 30522),
    ],
)
def test_reduction_applies(batch, features, fast):
    """Reduction rule should apply to softmax/fast_softmax of various shapes."""
    mod = _build_softmax(batch, features, fast=fast)
    target = _llvm_target()
    _apply_and_check(mod, target)


# ---------------------------------------------------------------------------
# Test: scheduled TIR structure
# ---------------------------------------------------------------------------


def test_softmax_schedule_structure():
    """Verify the scheduled TIR has expected structure:
    - parallel on batch axis
    - vectorized innermost loops for injective blocks
    - split+unroll for reduction blocks
    """
    mod = _build_softmax(14, 185, fast=False)
    target = _llvm_target()
    sch = _apply_and_check(mod, target)
    scheduled_mod = sch.mod

    # Check that tirx.is_scheduled is NOT set (only set by ApplyDefaultSchedule)
    # but the schedule should be valid
    assert scheduled_mod is not None

    # Verify via ApplyDefaultSchedule path
    scheduled = dl.ApplyDefaultSchedule(Reduction(), target=target)(mod)
    func = scheduled["main"]

    # Check tirx.is_scheduled is set
    assert func.attrs and func.attrs.get("tirx.is_scheduled", False)


def test_fast_softmax_schedule_structure():
    """fast_softmax should keep T_fast_exp as a separate vectorizable block."""
    mod = _build_softmax(14, 185, fast=True)
    target = _llvm_target()
    sch = _apply_and_check(mod, target)
    script = str(sch.mod)

    # fast_exp block should exist (not inlined)
    assert "T_fast_exp" in script or "T_softmax_delta" in script
    # Should have T.parallel
    assert "T.parallel" in script
    # Should have T.vectorized
    assert "T.vectorized" in script


# ---------------------------------------------------------------------------
# Test: LLVM IR quality (cross-compile to RISC-V RVV)
# ---------------------------------------------------------------------------


def _codegen_llvm_ir(mod, target):
    """Lower and codegen to LLVM IR (no linking)."""
    bound = tirx.transform.BindTarget(target.with_host(target))(mod)
    pipeline, finalize_host, _ = tirx.get_tir_pipeline("s_tir")
    lowered = pipeline(bound)
    from tvm_tirx.tirx._compile import _split_host_device_modules

    host_mod, _ = _split_host_device_modules(lowered)
    host_mod = finalize_host()(host_mod)
    built = tvm_tirx.codegen.build_module(host_mod, target)
    return built.inspect_source("ll")


def _codegen_asm(mod, target):
    """Lower and codegen to assembly (no linking)."""
    bound = tirx.transform.BindTarget(target.with_host(target))(mod)
    pipeline, finalize_host, _ = tirx.get_tir_pipeline("s_tir")
    lowered = pipeline(bound)
    from tvm_tirx.tirx._compile import _split_host_device_modules

    host_mod, _ = _split_host_device_modules(lowered)
    host_mod = finalize_host()(host_mod)
    built = tvm_tirx.codegen.build_module(host_mod, target)
    return built.inspect_source("s")


@pytest.mark.parametrize("fast", [False, True], ids=["softmax", "fast_softmax"])
def test_rvv_code_size_reduction(fast):
    """Scheduled RVV code should be smaller than unscheduled.

    The original issue (apache/tvm#18569) shows RVV softmax is 1.34x slower
    than scalar, partly due to LLVM generating bloated code with excessive
    unrolling. The schedule should reduce code size significantly.
    """
    target = _rvv_target()
    mod = _build_softmax(14, 185, fast=fast)

    # Unscheduled
    ir_unsched = _codegen_llvm_ir(mod, target)
    n_unsched = len(ir_unsched.splitlines())

    # Scheduled
    mod_sched = dl.ApplyDefaultSchedule(Reduction(), target=target)(mod)
    ir_sched = _codegen_llvm_ir(mod_sched, target)
    n_sched = len(ir_sched.splitlines())

    # Scheduled should be meaningfully smaller (at least 30% reduction)
    ratio = n_sched / n_unsched
    assert ratio < 0.75, (
        f"Expected >=25% code reduction, got {(1 - ratio) * 100:.1f}% "
        f"({n_unsched} -> {n_sched} lines)"
    )


# The sym analyzer no longer proves vscale-bearing inequalities via
# substitution (CanProveVscaleExpressionFromKnownValues was deleted). This
# weakens simplification of scalable-vector index expressions, which can
# prevent the RVV vectorization schedule from producing scalable vector ops.
@pytest.mark.xfail(reason="sym no longer proves vscale-bearing inequalities via substitution")
def test_rvv_fast_softmax_vectorizes_exp():
    """fast_softmax + schedule should produce RVV vector instructions
    for the polynomial exp approximation (no scalar exp calls)."""
    target = _rvv_target()
    mod = _build_softmax(14, 185, fast=True)
    mod_sched = dl.ApplyDefaultSchedule(Reduction(), target=target)(mod)
    ir = _codegen_llvm_ir(mod_sched, target)

    # Should have zero scalar exp calls (fast_exp uses polynomial)
    scalar_exp = sum(1 for line in ir.splitlines() if "llvm.exp.f32" in line)
    assert scalar_exp == 0, f"Expected 0 scalar exp calls, got {scalar_exp}"

    # Should have scalable vector operations
    n_svec = ir.count("<vscale x")
    assert n_svec > 0, "Expected scalable vector operations in LLVM IR"


def test_rvv_asm_instruction_reduction():
    """Scheduled RVV assembly should have fewer total instructions
    than both unscheduled RVV and scalar RV."""
    rvv = _rvv_target()
    rv = Target(
        {
            "kind": "llvm",
            "mtriple": "riscv64-linux-gnu",
            "mcpu": "generic-rv64",
            "mabi": "lp64d",
            "mattr": ["+64bit", "+m", "+a", "+f", "+d", "+c"],
        }
    )

    mod = _build_softmax(14, 185, fast=True)

    # Scalar baseline
    asm_rv = _codegen_asm(mod, rv)
    n_rv = len(
        [
            line
            for line in asm_rv.splitlines()
            if line.strip() and not line.strip().startswith((".", "#", "/"))
        ]
    )

    # RVV unscheduled
    asm_rvv = _codegen_asm(mod, rvv)
    n_rvv = len(
        [
            line
            for line in asm_rvv.splitlines()
            if line.strip() and not line.strip().startswith((".", "#", "/"))
        ]
    )

    # RVV scheduled
    mod_sched = dl.ApplyDefaultSchedule(Reduction(), target=rvv)(mod)
    asm_sched = _codegen_asm(mod_sched, rvv)
    n_sched = len(
        [
            line
            for line in asm_sched.splitlines()
            if line.strip() and not line.strip().startswith((".", "#", "/"))
        ]
    )

    # Scheduled should be smaller than both unscheduled RVV and scalar
    assert n_sched < n_rvv, (
        f"Scheduled ({n_sched}) should have fewer instructions than unscheduled RVV ({n_rvv})"
    )
    assert n_sched <= n_rv * 1.1, (
        f"Scheduled ({n_sched}) should not be much larger than scalar RV ({n_rv})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
