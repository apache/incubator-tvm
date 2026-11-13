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
"""S-TIR script namespace."""

import os
import subprocess
import sys

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_shared_operations_and_aliases():
    from tvm.s_tir import script as S

    n = S.dynamic("n")

    @S.function
    def shared(A: S.Tensor((n,), "float32")):
        for i in T.serial(A.shape[0]):
            with S.sblock("copy"):
                v = Ts.axis.spatial(A.shape[0], i)
                A[v] = S.float32(1)

    assert shared.attrs["script.namespace"] == "s_tir"
    assert shared.params[0].ty.layout is None
    assert Ts.Tensor is T.Tensor
    assert Ts.serial is T.serial
    assert Ts.Bind is T.Bind
    assert Ts.tile is T.tile
    tvm_tirx.ir.assert_structural_equal(
        shared,
        tvm_tirx.script.from_source(
            shared.script(),
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
        ),
    )


def test_mixed_module_roundtrip():
    @I.ir_module
    class Mixed:
        @Ts.function
        def scheduled(A: Ts.Tensor((4,), "float32")):
            for i in Ts.serial(4):
                with Ts.sblock("copy"):
                    v = Ts.axis.spatial(4, i)
                    A[v] = 1.0

        @T.function
        def direct(A: T.Tensor((4,), "float32")):
            for i in T.serial(4):
                A[i] = 2.0

    assert Mixed["scheduled"].attrs["script.namespace"] == "s_tir"
    assert Mixed["direct"].attrs.get("script.namespace") != "s_tir"
    script = Mixed.script()
    assert "@Ts.function" in script
    assert "@T.function" in script
    assert "s_tir=True" not in script
    tvm_tirx.ir.assert_structural_equal(
        Mixed,
        tvm_tirx.script.from_source(
            script,
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
        ),
    )
    direct_script = Mixed["direct"].script()
    assert "Ts" not in direct_script
    assert "import s_tir" not in direct_script


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM is not enabled")
def test_tirx_construction_roundtrip_and_execution_are_independent(monkeypatch):
    def reject_s_tir_analysis(*args, **kwargs):
        pytest.fail("TIRx construction and compilation must not invoke S-TIR verification")

    monkeypatch.setattr(tvm.s_tir.analysis, "check_well_formed", reject_s_tir_analysis)

    @I.ir_module
    class Direct:
        @T.function
        def main(A: T.Tensor((4,), "int32")):
            for i in T.serial(4):
                A[i] = A[i] + 3

    restored = tvm_tirx.script.from_source(
        Direct.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
    )
    tvm_tirx.ir.assert_structural_equal(Direct, restored)
    assert "s_tir" not in restored["main"].attrs
    executable = tvm.compile(restored, target="llvm")
    data = tvm_tirx.runtime.tensor(np.arange(4, dtype="int32"))
    executable["main"](data)
    np.testing.assert_array_equal(data.numpy(), np.arange(4, dtype="int32") + 3)


@pytest.mark.parametrize("namespace", ["Ts"])
@pytest.mark.parametrize("option", ["s_tir", "is_stir"])
def test_legacy_mode_is_not_a_function_option(namespace, option):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        tvm_tirx.script.from_source(
            f"@{namespace}.function({option}=True, check_well_formed=False)\n"
            "def main():\n    T.evaluate(0)\n",
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
        )


@pytest.mark.parametrize(
    "operation",
    [
        "sblock",
        "init",
        "where",
        "reads",
        "writes",
        "sblock_attr",
        "sblock_alloc_tensor",
        "axis",
        "block_name_suffix_context",
    ],
)
def test_s_tir_operations_have_an_independent_namespace(operation):
    from tvm.s_tir.script import ir_builder as s_tir_builder
    from tvm_tirx.tirx.script import ir_builder as tirx_builder

    assert hasattr(Ts, operation)
    assert hasattr(s_tir_builder, operation)
    assert not hasattr(T, operation)
    assert not hasattr(tirx_builder, operation)


def test_tirx_rejects_s_tir_blocks():
    with pytest.raises(ValueError, match="Ts.function"):
        tvm_tirx.script.from_source(
            "@T.function(check_well_formed=False)\n"
            "def main():\n    with Ts.sblock('bad'):\n        T.evaluate(0)\n",
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
        )


def test_s_tir_options_and_helpers():
    @Ts.inline
    def fill(A):
        for i in Ts.serial(4):
            A[i] = Ts.float32(1)

    @Ts.function(private=True)
    def scheduled(A: Ts.Tensor((4,), "float32")):
        fill(A)

    assert scheduled.attrs["script.namespace"] == "s_tir"
    assert "global_symbol" not in scheduled.attrs
    tvm_tirx.ir.assert_structural_equal(
        scheduled,
        tvm_tirx.script.from_source(
            scheduled.script(),
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
        ),
    )


@pytest.mark.parametrize(
    "first",
    [
        "tvm.s_tir.script",
        "tvm_tirx.tirx.script",
        "tvm_tirx.script.parser",
        "tvm.s_tir.script.ir_builder.frame",
        "tvm_tirx.script.ir_builder.s_tir.frame",
    ],
)
def test_import_order(first):
    # Parent initialization registers its deep builder alias with the shared core.
    setup = "import tvm\n" if first == "tvm_tirx.script.ir_builder.s_tir.frame" else ""
    subprocess.run(
        [
            sys.executable,
            "-c",
            setup + f"import {first}\n"
            "import tvm.s_tir.script as direct\n"
            "from tvm.s_tir import script as Ts\nfrom tvm_tirx.script import tirx as T\n"
            "from tvm_tirx.script.parser import s_tir as parser\n"
            "assert Ts is direct is parser\n"
            "assert Ts.Tensor is T.Tensor\n"
            "from tvm.s_tir.script import ir_builder as owned\n"
            "from tvm_tirx.script.ir_builder import s_tir as alias\n"
            "import tvm.s_tir.script.ir_builder.frame as owned_frame\n"
            "import tvm_tirx.script.ir_builder.s_tir.frame as alias_frame\n"
            "assert Ts.ir_builder is owned is alias\n"
            "assert owned.__name__ == 'tvm.s_tir.script.ir_builder'\n"
            "assert owned_frame is alias_frame\n"
            "assert Ts.ir_builder is not T.ir_builder\n"
            "from tvm_tirx.script.parser import _NAMESPACES\n"
            "assert _NAMESPACES['s_tir'] is Ts\n"
            "assert _NAMESPACES['tirx'] is T\n",
        ],
        check=True,
        env=os.environ.copy(),
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
