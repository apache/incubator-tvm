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
# pylint: disable=invalid-name,missing-function-docstring
"""Dot product related intrinsics."""

from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T

from .. import TensorIntrin


def get_dp4a_intrin(dtype_a, dtype_b, dtype_c):
    if dtype_c == "uint32":
        assert dtype_a == dtype_b == "uint8"

    dp4a_desc_a_offset = T.dynamic("dp4a_desc_a_offset", "int32")
    dp4a_desc_b_offset = T.dynamic("dp4a_desc_b_offset", "int32")
    dp4a_desc_c_offset = T.dynamic("dp4a_desc_c_offset", "int32")

    @Ts.function
    def dp4a_desc(
        A: T.Tensor(
            (4,),
            dtype_a,
            layout=T.TileLayout(T.S[(4,)] + (dp4a_desc_a_offset)),
            align=4,
            scope="shared",
        ),
        B: T.Tensor(
            (4,),
            dtype_b,
            layout=T.TileLayout(T.S[(4,)] + (dp4a_desc_b_offset)),
            align=4,
            scope="shared",
        ),
        C: T.Tensor(
            (1,),
            dtype_c,
            layout=T.TileLayout(T.S[(1,)] + (dp4a_desc_c_offset)),
            align=4,
            scope="local",
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0], A[0:4], B[0:4])
            Ts.writes(C[0])
            for i in range(0, 4):
                with Ts.sblock("update"):
                    vi = Ts.axis.remap("R", [i])
                    C[0] = C[0] + T.cast(A[vi], ty=dtype_c) * T.cast(B[vi], ty=dtype_c)

    dp4a_impl_a_offset = T.dynamic("dp4a_impl_a_offset", "int32")
    dp4a_impl_b_offset = T.dynamic("dp4a_impl_b_offset", "int32")
    dp4a_impl_c_offset = T.dynamic("dp4a_impl_c_offset", "int32")

    @Ts.function
    def dp4a_impl(
        A: T.Tensor(
            (4,),
            dtype_a,
            layout=T.TileLayout(T.S[(4,)] + (dp4a_impl_a_offset)),
            align=4,
            scope="shared",
        ),
        B: T.Tensor(
            (4,),
            dtype_b,
            layout=T.TileLayout(T.S[(4,)] + (dp4a_impl_b_offset)),
            align=4,
            scope="shared",
        ),
        C: T.Tensor(
            (1,),
            dtype_c,
            layout=T.TileLayout(T.S[(1,)] + (dp4a_impl_c_offset)),
            align=4,
            scope="local",
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0], A[0:4], B[0:4])
            Ts.writes(C[0])

            C[0] += T.call_pure_extern(
                "__dp4a",
                A[T.Ramp(0, 1, 4)],
                B[T.Ramp(0, 1, 4)],
                T.uint32(0) if T.constexpr(dtype_c == "uint32") else T.int32(0),
                ty=dtype_c,
            )

    return dp4a_desc, dp4a_impl


DP4A_S8S8S32_INTRIN = "dp4a_s8s8s32"
TensorIntrin.register(DP4A_S8S8S32_INTRIN, *get_dp4a_intrin("int8", "int8", "int32"))
DP4A_U8S8S32_INTRIN = "dp4a_u8s8s32"
TensorIntrin.register(DP4A_U8S8S32_INTRIN, *get_dp4a_intrin("uint8", "int8", "int32"))
DP4A_S8U8S32_INTRIN = "dp4a_s8u8s32"
TensorIntrin.register(DP4A_S8U8S32_INTRIN, *get_dp4a_intrin("int8", "uint8", "int32"))
DP4A_U8U8U32_INTRIN = "dp4a_u8u8u32"
TensorIntrin.register(DP4A_U8U8U32_INTRIN, *get_dp4a_intrin("uint8", "uint8", "uint32"))
