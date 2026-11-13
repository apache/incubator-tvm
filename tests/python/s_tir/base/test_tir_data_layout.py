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
"""Test layout and bijective-layout node"""

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.topi.utils import get_const_tuple


def test_layout():
    layout = tvm.s_tir.slayout("NCHW16c")
    assert layout is not None
    assert isinstance(layout, tvm.s_tir.SLayout)

    assert layout.factor_of("c") == 16
    assert layout.factor_of("C") == 16
    assert layout.factor_of("N") == -1

    assert layout.index_of("N") == 0
    assert layout.index_of("C") == 1
    assert layout.index_of("H") == 2
    assert layout.index_of("W") == 3
    assert layout.index_of("16c") == 4
    assert layout.index_of("O") == -1

    assert "N" in layout
    assert "C" in layout
    assert "H" in layout
    assert "W" in layout
    assert "c" in layout
    assert "O" not in layout

    assert layout[0] == "N"
    assert layout[1] == "C"
    assert layout[2] == "H"
    assert layout[3] == "W"
    assert layout[4] == "16c"

    layout = tvm.s_tir.slayout("OIHW[4o4i]")
    assert layout is not None
    assert isinstance(layout, tvm.s_tir.SLayout)

    assert layout.factor_of("o") == 4
    assert layout.factor_of("i") == 4
    assert layout.factor_of("H") == -1
    assert layout.factor_of("W") == -1
    assert layout.factor_of("N") == -1

    assert layout.index_of("O") == 0
    assert layout.index_of("I") == 1
    assert layout.index_of("H") == 2
    assert layout.index_of("W") == 3
    assert layout.index_of("4o4i") == 4
    assert layout.index_of("i") == -1
    assert layout.index_of("o") == -1

    assert "O" in layout
    assert "I" in layout
    assert "H" in layout
    assert "W" in layout
    assert "4o4i" in layout
    assert "i" in layout
    assert "o" in layout

    assert layout[0] == "O"
    assert layout[1] == "I"
    assert layout[2] == "H"
    assert layout[3] == "W"
    assert layout[4] == "4o4i"

    with pytest.raises(RuntimeError):
        layout = tvm.s_tir.slayout("[N4o]C")
    with pytest.raises(RuntimeError):
        layout = tvm.s_tir.slayout("[O4o]")
    with pytest.raises(RuntimeError):
        layout = tvm.s_tir.slayout("C4o")
    with pytest.raises(RuntimeError):
        layout = tvm.s_tir.slayout("OI[4o4i][]")
    with pytest.raises(RuntimeError):
        layout = tvm.s_tir.slayout("C4c[4c]")


def test_layout_dtype():
    layout_i32 = tvm.s_tir.slayout("NCHW")
    assert layout_i32.axes[0].var.ty.dtype == "int32"
    assert layout_i32.axes[0].dom.begin.ty.dtype == "int32"
    assert layout_i32.axes[0].dom.extent.ty.dtype == "int32"
    assert layout_i32.axes[1].var.ty.dtype == "int32"
    assert layout_i32.axes[1].dom.begin.ty.dtype == "int32"
    assert layout_i32.axes[1].dom.extent.ty.dtype == "int32"

    layout_i64 = tvm.s_tir.slayout("NCHW", dtype="int64")
    assert layout_i64.axes[2].var.ty.dtype == "int64"
    assert layout_i64.axes[2].dom.begin.ty.dtype == "int64"
    assert layout_i64.axes[2].dom.extent.ty.dtype == "int64"
    assert layout_i64.axes[3].var.ty.dtype == "int64"
    assert layout_i64.axes[3].dom.begin.ty.dtype == "int64"
    assert layout_i64.axes[3].dom.extent.ty.dtype == "int64"

    with pytest.raises(TypeError):
        tvm.s_tir.slayout("NCHW", dtype="float32")
    with pytest.raises(TypeError):
        tvm.s_tir.slayout("NCHW", dtype=None)


def test_bilayout_convertible():
    # not convertible
    assert tvm.s_tir.sbijective_layout("NCHW", "ABCD") is None
    assert tvm.s_tir.sbijective_layout("__undef__", "NCHW") is None
    assert tvm.s_tir.sbijective_layout("NCHW", "__undef__") is None
    assert tvm.s_tir.sbijective_layout("__undef__", "__undef__") is None
    assert tvm.s_tir.sbijective_layout("", "NCHW") is None
    assert tvm.s_tir.sbijective_layout("NCHW", "") is None
    assert tvm.s_tir.sbijective_layout("OIHW", "OIHW[4o4i]") is not None
    assert tvm.s_tir.sbijective_layout("OIHW[2o4i]", "OIHW") is not None
    assert tvm.s_tir.sbijective_layout("", "") is None
    # convertible
    assert tvm.s_tir.sbijective_layout("NCHW", "NCHW16c") is not None


def test_bilayout_shape():
    bilayout = tvm.s_tir.sbijective_layout("NCHW", "NCHW16c")
    assert isinstance(bilayout, tvm.s_tir.SBijectiveLayout)

    dst_shape = bilayout.forward_shape((1, 32, 7, 7))
    assert get_const_tuple(dst_shape) == (1, 2, 7, 7, 16)

    src_shape = bilayout.backward_shape(dst_shape)
    assert get_const_tuple(src_shape) == (1, 32, 7, 7)

    bilayout = tvm.s_tir.sbijective_layout("OIHW", "OIHW[4o4i]")

    dst_shape = bilayout.forward_shape((64, 28, 7, 7))
    assert get_const_tuple(dst_shape) == (16, 7, 7, 7, 16)

    src_shape = bilayout.backward_shape((2, 11, 4, 4, 16))
    assert get_const_tuple(src_shape) == (8, 44, 4, 4)


def test_bilayout_index():
    bilayout = tvm.s_tir.sbijective_layout("NCHW", "NCHW16c")

    dst_index = bilayout.forward_index([0, 18, 6, 6])
    assert get_const_tuple(dst_index) == (0, 1, 6, 6, 2)

    src_index = bilayout.backward_index([0, 1, 6, 6, 2])
    assert get_const_tuple(src_index) == (0, 18, 6, 6)

    bilayout = tvm.s_tir.sbijective_layout("OIHW", "OIHW[4o4i]")

    dst_index = bilayout.forward_index((63, 29, 7, 7))
    assert get_const_tuple(dst_index) == (15, 7, 7, 7, 13)

    src_index = bilayout.backward_index((4, 7, 4, 4, 13))
    assert get_const_tuple(src_index) == (19, 29, 4, 4)


if __name__ == "__main__":
    tvm_tirx.testing.main()
