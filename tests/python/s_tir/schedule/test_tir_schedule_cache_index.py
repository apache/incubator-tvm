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
# pylint: disable=missing-function-docstring,missing-module-docstring
# ruff: noqa: F401
import sys

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import verify_trace_roundtrip
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# pylint: disable=no-member,invalid-name,unused-variable

########## Function before schedule ##########


@Ts.function
def resize(A: T.Tensor((1, 3, 40, 40), "float32"), B: T.Tensor((1, 3, 80, 80), "float32")) -> None:
    for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
        with Ts.sblock("A"):
            n, c, vi, vj = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            B[n, c, vi, vj] = A[n, c, vi // 4 + vj // 4, vj // 2]


@Ts.function
def resize_cache_index(
    A: T.Tensor((1, 3, 40, 40), "float32"), B: T.Tensor((1, 3, 80, 80), "float32")
) -> None:
    index_var_0 = Ts.sblock_alloc_tensor(
        [80, 80], dtype="int32", layout=T.TileLayout(T.S[[80, 80] : [80, 1]])
    )
    index_var_1 = Ts.sblock_alloc_tensor([80], dtype="int32", layout=T.TileLayout(T.S[[80] : [1]]))
    for ax0, ax1 in T.grid(80, 80):
        with Ts.sblock("index_0"):
            v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
            Ts.reads()
            Ts.writes(index_var_0[v0, v1])
            index_var_0[v0, v1] = v0 // 4 + v1 // 4
    for ax0 in T.serial(80):
        with Ts.sblock("index_1"):
            v0 = Ts.axis.spatial(80, ax0)
            Ts.reads()
            Ts.writes(index_var_1[v0])
            index_var_1[v0] = v0 // 2
    for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
        with Ts.sblock("A"):
            n, c, vi, vj = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            Ts.reads(A[n, c, vi // 4 + vj // 4, vj // 2])
            Ts.writes(B[n, c, vi, vj])
            B[n, c, vi, vj] = A[n, c, index_var_0[vi, vj], index_var_1[vj]]


@Ts.function
def bilinear_resize(
    x: T.Tensor((1, 3, 40, 40), "float16"), resize: T.Tensor((1, 3, 80, 80), "float16")
):
    for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
        with Ts.sblock("resize"):
            i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            Ts.reads(x[i0_1, i1_1, 0:40, 0:40])
            Ts.writes(resize[i0_1, i1_1, i2_1, i3_1])
            resize[i0_1, i1_1, i2_1, i3_1] = T.Cast(
                "float16",
                (
                    T.Cast(
                        "float32",
                        x[
                            i0_1,
                            i1_1,
                            T.max(
                                T.min(
                                    T.Cast(
                                        "int32",
                                        T.floor(
                                            (T.Cast("float32", i2_1) + T.float32(0.5))
                                            * T.float32(0.5)
                                            - T.float32(0.5),
                                        ),
                                    ),
                                    39,
                                ),
                                0,
                            ),
                            T.max(
                                T.min(
                                    T.Cast(
                                        "int32",
                                        T.floor(
                                            (T.Cast("float32", i3_1) + T.float32(0.5))
                                            * T.float32(0.5)
                                            - T.float32(0.5),
                                        ),
                                    ),
                                    39,
                                ),
                                0,
                            ),
                        ],
                    )
                    * (
                        T.float32(1)
                        - (
                            (T.Cast("float32", i3_1) + T.float32(0.5)) * T.float32(0.5)
                            - T.float32(0.5)
                            - T.Cast(
                                "float32",
                                T.Cast(
                                    "int32",
                                    T.floor(
                                        (T.Cast("float32", i3_1) + T.float32(0.5)) * T.float32(0.5)
                                        - T.float32(0.5),
                                    ),
                                ),
                            )
                        )
                    )
                    + T.Cast(
                        "float32",
                        x[
                            i0_1,
                            i1_1,
                            T.max(
                                T.min(
                                    T.Cast(
                                        "int32",
                                        T.floor(
                                            (T.Cast("float32", i2_1) + T.float32(0.5))
                                            * T.float32(0.5)
                                            - T.float32(0.5),
                                        ),
                                    ),
                                    39,
                                ),
                                0,
                            ),
                            T.max(
                                T.min(
                                    T.Cast(
                                        "int32",
                                        T.floor(
                                            (T.Cast("float32", i3_1) + T.float32(0.5))
                                            * T.float32(0.5)
                                            - T.float32(0.5),
                                        ),
                                    )
                                    + 1,
                                    39,
                                ),
                                0,
                            ),
                        ],
                    )
                    * (
                        (T.Cast("float32", i3_1) + T.float32(0.5)) * T.float32(0.5)
                        - T.float32(0.5)
                        - T.Cast(
                            "float32",
                            T.Cast(
                                "int32",
                                T.floor(
                                    (T.Cast("float32", i3_1) + T.float32(0.5)) * T.float32(0.5)
                                    - T.float32(0.5),
                                ),
                            ),
                        )
                    )
                )
                * (
                    T.float32(1)
                    - (
                        (T.Cast("float32", i2_1) + T.float32(0.5)) * T.float32(0.5)
                        - T.float32(0.5)
                        - T.Cast(
                            "float32",
                            T.Cast(
                                "int32",
                                T.floor(
                                    (T.Cast("float32", i2_1) + T.float32(0.5)) * T.float32(0.5)
                                    - T.float32(0.5),
                                ),
                            ),
                        )
                    )
                )
                + (
                    T.Cast(
                        "float32",
                        x[
                            i0_1,
                            i1_1,
                            T.max(
                                T.min(
                                    T.Cast(
                                        "int32",
                                        T.floor(
                                            (T.Cast("float32", i2_1) + T.float32(0.5))
                                            * T.float32(0.5)
                                            - T.float32(0.5),
                                        ),
                                    )
                                    + 1,
                                    39,
                                ),
                                0,
                            ),
                            T.max(
                                T.min(
                                    T.Cast(
                                        "int32",
                                        T.floor(
                                            (T.Cast("float32", i3_1) + T.float32(0.5))
                                            * T.float32(0.5)
                                            - T.float32(0.5),
                                        ),
                                    ),
                                    39,
                                ),
                                0,
                            ),
                        ],
                    )
                    * (
                        T.float32(1)
                        - (
                            (T.Cast("float32", i3_1) + T.float32(0.5)) * T.float32(0.5)
                            - T.float32(0.5)
                            - T.Cast(
                                "float32",
                                T.Cast(
                                    "int32",
                                    T.floor(
                                        (T.Cast("float32", i3_1) + T.float32(0.5)) * T.float32(0.5)
                                        - T.float32(0.5),
                                    ),
                                ),
                            )
                        )
                    )
                    + T.Cast(
                        "float32",
                        x[
                            i0_1,
                            i1_1,
                            T.max(
                                T.min(
                                    T.Cast(
                                        "int32",
                                        T.floor(
                                            (T.Cast("float32", i2_1) + T.float32(0.5))
                                            * T.float32(0.5)
                                            - T.float32(0.5),
                                        ),
                                    )
                                    + 1,
                                    39,
                                ),
                                0,
                            ),
                            T.max(
                                T.min(
                                    T.Cast(
                                        "int32",
                                        T.floor(
                                            (T.Cast("float32", i3_1) + T.float32(0.5))
                                            * T.float32(0.5)
                                            - T.float32(0.5),
                                        ),
                                    )
                                    + 1,
                                    39,
                                ),
                                0,
                            ),
                        ],
                    )
                    * (
                        (T.Cast("float32", i3_1) + T.float32(0.5)) * T.float32(0.5)
                        - T.float32(0.5)
                        - T.Cast(
                            "float32",
                            T.Cast(
                                "int32",
                                T.floor(
                                    (T.Cast("float32", i3_1) + T.float32(0.5)) * T.float32(0.5)
                                    - T.float32(0.5),
                                ),
                            ),
                        )
                    )
                )
                * (
                    (T.Cast("float32", i2_1) + T.float32(0.5)) * T.float32(0.5)
                    - T.float32(0.5)
                    - T.Cast(
                        "float32",
                        T.Cast(
                            "int32",
                            T.floor(
                                (T.Cast("float32", i2_1) + T.float32(0.5)) * T.float32(0.5)
                                - T.float32(0.5),
                            ),
                        ),
                    )
                ),
            )


@Ts.function
def cached_bilinear_resize(
    x: T.Tensor((1, 3, 40, 40), "float16"), resize: T.Tensor((1, 3, 80, 80), "float16")
):
    index_var_0 = Ts.sblock_alloc_tensor(
        [80], dtype="float32", layout=T.TileLayout(T.S[[80] : [1]])
    )
    index_var_1 = Ts.sblock_alloc_tensor([80], dtype="int32", layout=T.TileLayout(T.S[[80] : [1]]))
    index_var_2 = Ts.sblock_alloc_tensor([80], dtype="int32", layout=T.TileLayout(T.S[[80] : [1]]))
    for ax0 in T.serial(80):
        with Ts.sblock("index_0"):
            v0 = Ts.axis.spatial(80, ax0)
            Ts.reads()
            Ts.writes(index_var_0[v0])
            index_var_0[v0] = (
                (T.Cast("float32", v0) + T.float32(0.5)) * T.float32(0.5)
                - T.float32(0.5)
                - T.Cast(
                    "float32",
                    T.Cast(
                        "int32",
                        T.floor(
                            (T.Cast("float32", v0) + T.float32(0.5)) * T.float32(0.5)
                            - T.float32(0.5),
                        ),
                    ),
                )
            )
    for ax0 in T.serial(80):
        with Ts.sblock("index_1"):
            v0 = Ts.axis.spatial(80, ax0)
            Ts.reads()
            Ts.writes(index_var_1[v0])
            index_var_1[v0] = T.Cast(
                "int32",
                T.floor(
                    (T.Cast("float32", v0) + T.float32(0.5)) * T.float32(0.5) - T.float32(0.5),
                ),
            )
    for ax0 in T.serial(80):
        with Ts.sblock("index_2"):
            v0 = Ts.axis.spatial(80, ax0)
            Ts.reads()
            Ts.writes(index_var_2[v0])
            index_var_2[v0] = T.Cast(
                "int32",
                T.floor(
                    (T.Cast("float32", v0) + T.float32(0.5)) * T.float32(0.5) - T.float32(0.5),
                ),
            )
    for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
        with Ts.sblock("resize"):
            i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            Ts.reads(x[i0_1, i1_1, 0:40, 0:40])
            Ts.writes(resize[i0_1, i1_1, i2_1, i3_1])
            resize[i0_1, i1_1, i2_1, i3_1] = T.Cast(
                "float16",
                (
                    T.Cast(
                        "float32",
                        x[
                            i0_1,
                            i1_1,
                            T.max(T.min(index_var_1[i2_1], 39), 0),
                            T.max(T.min(index_var_2[i3_1], 39), 0),
                        ],
                    )
                    * (T.float32(1) - index_var_0[i3_1])
                    + T.Cast(
                        "float32",
                        x[
                            i0_1,
                            i1_1,
                            T.max(T.min(index_var_1[i2_1], 39), 0),
                            T.max(T.min(index_var_2[i3_1] + 1, 39), 0),
                        ],
                    )
                    * index_var_0[i3_1]
                )
                * (
                    T.float32(1)
                    - (
                        (T.Cast("float32", i2_1) + T.float32(0.5)) * T.float32(0.5)
                        - T.float32(0.5)
                        - T.Cast("float32", index_var_1[i2_1])
                    )
                )
                + (
                    T.Cast(
                        "float32",
                        x[
                            i0_1,
                            i1_1,
                            T.max(T.min(index_var_1[i2_1] + 1, 39), 0),
                            T.max(T.min(index_var_2[i3_1], 39), 0),
                        ],
                    )
                    * (T.float32(1) - index_var_0[i3_1])
                    + T.Cast(
                        "float32",
                        x[
                            i0_1,
                            i1_1,
                            T.max(T.min(index_var_1[i2_1] + 1, 39), 0),
                            T.max(T.min(index_var_2[i3_1] + 1, 39), 0),
                        ],
                    )
                    * index_var_0[i3_1]
                )
                * (
                    (T.Cast("float32", i2_1) + T.float32(0.5)) * T.float32(0.5)
                    - T.float32(0.5)
                    - T.Cast("float32", index_var_1[i2_1])
                ),
            )


def test_basic_cache_index():
    sch = tvm.s_tir.Schedule(resize, debug_mask="all")
    block = sch.get_sblock("A")
    sch.cache_index(block, "global")
    tvm_tirx.ir.assert_structural_equal(
        resize_cache_index, sch.mod["main"].with_attr("global_symbol", "resize_cache_index")
    )
    verify_trace_roundtrip(sch=sch, mod=resize)


def test_resize_bilinear_cache_index():
    sch = tvm.s_tir.Schedule(bilinear_resize, debug_mask="all")
    block = sch.get_sblock("resize")
    sch.cache_index(block, "global", 4)
    tvm_tirx.ir.assert_structural_equal(
        sch.mod["main"], cached_bilinear_resize.with_attr("global_symbol", "bilinear_resize")
    )
    verify_trace_roundtrip(sch=sch, mod=bilinear_resize)


if __name__ == "__main__":
    tvm_tirx.testing.main()
