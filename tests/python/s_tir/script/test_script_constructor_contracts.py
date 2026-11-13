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

"""S-TIR constructor contracts and printer namespace integration."""

import pytest
from tvm_ffi import get_global_func

import tvm_tirx
from tvm import s_tir
from tvm.s_tir import script as S
from tvm_tirx import ir, tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.script import ir_builder as TB


def _assert_print(obj, expected):
    assert obj.script(verbose_expr=True).strip() == expected.strip()


def test_cast_script_positional_and_keyword_type():
    @S.function(private=True)
    def positional(x: S.int32):
        S.evaluate(S.cast(x, "float32"))

    @S.function(private=True)
    def keyword(x: S.int32):
        S.evaluate(S.cast(x, ty="float32"))

    assert S.cast is T.cast
    assert "s_tir" not in positional.attrs
    assert positional.attrs["script.namespace"] == "s_tir"
    ir.assert_structural_equal(positional, keyword)
    cast_expr = positional.body[0].value
    assert cast_expr.ty == ir.PrimType("float32")
    assert isinstance(cast_expr.loc, ir.SourceLoc)
    ir.assert_structural_equal(
        positional, tvm_tirx.script.from_source(positional.script(), extra_vars={"T": T, "Ts": S})
    )


def test_iter_var():
    a = s_tir.IterVar((0, 8), "a", iter_type=s_tir.IterVar.DataPar)
    _assert_print(
        a,
        """
a = T.dynamic("a", dtype="int32")
Ts.iter_var(a, T.Range(0, 8), "DataPar", "")
""",
    )


def test_comm_reducer():
    from tvm.s_tir.script import ir_builder as SB

    obj = SB.comm_reducer(lambda x, y: x + y, identity=[TB.float32(0)])
    _assert_print(
        obj,
        """
Ts.comm_reducer(lambda x, y: x + y, [T.float32(0.0)])
""",
    )


@pytest.mark.parametrize("value", ["2prefix", 17])
@pytest.mark.parametrize("nested", [False, True])
def test_config_validates_s_tir_prefix(value, nested):
    config = {"s_tir.prefix": value}
    if nested:
        config = {"extra_config": config}
    with pytest.raises((RuntimeError, TypeError)):
        get_global_func("node.PrinterConfig")(config)


@pytest.mark.parametrize("prefixes", [{}, {"tirx.prefix": "CustomT", "s_tir.prefix": "CustomTs"}])
def test_config_reserves_s_tir_prefix_before_variable_definition(prefixes):
    tir_prefix = prefixes.get("tirx.prefix", "T")
    name = prefixes.get("s_tir.prefix", "Ts")
    var = tirx.Var(name, "int32")
    assert var.script(verbose_expr=True, extra_config=prefixes).strip() == (
        f'{name}_1 = {tir_prefix}.dynamic("{name}", dtype="int32")\n{name}_1'
    )
