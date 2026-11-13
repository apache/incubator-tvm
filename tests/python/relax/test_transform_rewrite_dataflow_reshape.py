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
# ruff: noqa: E501, F841

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_reshape_expand_dims():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def reshape(
            rxplaceholder: T.Tensor((T.int64(8), T.int64(3)), "float32"),
            T_reshape: T.Tensor((T.int64(2), T.int64(4), T.int64(3)), "float32"),
        ):
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(3)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(
                        rxplaceholder[
                            (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                            (v_ax0 * 12 + v_ax1 * 3 + v_ax2) % T.int64(3),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) % T.int64(3),
                    ]

        @Ts.function
        def expand_dims(
            rxplaceholder: T.Tensor((T.int64(2), T.int64(4), T.int64(3)), "float32"),
            expand_dims: T.Tensor(
                (T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)), "float32"
            ),
        ):
            for i0, i1, i2, i3, i4 in T.grid(
                T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)
            ):
                with Ts.sblock("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    Ts.reads(rxplaceholder[i0_1, i2_1, i4_1])
                    Ts.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1] = rxplaceholder[i0_1, i2_1, i4_1]

        @R.function
        def main(x: R.Tensor((8, 3), dtype="float32")) -> R.Tensor(
            (2, 1, 4, 1, 3), dtype="float32"
        ):
            cls = Module
            with R.dataflow():
                y = R.call_tir(cls.reshape, (x,), ty_args=[R.Tensor((2, 4, 3), dtype="float32")])
                z = R.call_tir(
                    cls.expand_dims, (y,), ty_args=[R.Tensor((2, 1, 4, 1, 3), "float32")]
                )
                R.output(z)
            return z

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def reshape(
            rxplaceholder: T.Tensor((T.int64(8), T.int64(3)), "float32"),
            T_reshape: T.Tensor((T.int64(2), T.int64(4), T.int64(3)), "float32"),
        ):
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(3)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(
                        rxplaceholder[
                            (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) // T.int64(3),
                            (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) % T.int64(3),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) // T.int64(3),
                        (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) % T.int64(3),
                    ]

        @Ts.function
        def expand_dims(
            rxplaceholder: T.Tensor((T.int64(2), T.int64(4), T.int64(3)), "float32"),
            expand_dims: T.Tensor(
                (T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)), "float32"
            ),
        ):
            for i0, i1, i2, i3, i4 in T.grid(
                T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)
            ):
                with Ts.sblock("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    Ts.reads(rxplaceholder[i0_1, i2_1, i4_1])
                    Ts.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1] = rxplaceholder[i0_1, i2_1, i4_1]

        @R.function
        def main(x: R.Tensor((8, 3), dtype="float32")) -> R.Tensor(
            (2, 1, 4, 1, 3), dtype="float32"
        ):
            with R.dataflow():
                cls = Expected
                y: R.Tensor((2, 4, 3), "float32") = R.reshape(x, (2, 4, 3))
                # Note: `z` is the output var of the dataflow block, and is thus
                # not expected to be rewritten.
                z = R.call_tir(
                    cls.expand_dims, (y,), ty_args=[R.Tensor((2, 1, 4, 1, 3), dtype="float32")]
                )
                R.output(z)
            return z

    assert relax.analysis.has_reshape_pattern(Module["expand_dims"])
    mod = relax.transform.RewriteDataflowReshape()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_reshape_pattern_detect():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def reshape(rxplaceholder: T.Tensor((T.int64(2), T.int64(4096), T.int64(320)), "float32"), T_reshape: T.Tensor((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32")):
            for ax0_ax1_ax2_ax3_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "blockIdx.x"}):
                for ax0_ax1_ax2_ax3_fused_2 in T.parallel(T.int64(1024), annotations={"thread_binding": "threadIdx.x"}):
                    for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(10)):
                        with Ts.sblock("T_reshape"):
                            v_ax0 = Ts.axis.spatial(T.int64(2), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(1310720))
                            v_ax1 = Ts.axis.spatial(T.int64(4096), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(1310720) // T.int64(320))
                            v_ax2 = Ts.axis.spatial(T.int64(5), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(320) // T.int64(64))
                            v_ax3 = Ts.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(64))
                            Ts.reads(rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)])
                            Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)]

        @Ts.function
        def expand_dims(
            rxplaceholder: T.Tensor((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32"),
            expand_dims: T.Tensor(
                (T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)),
                "float32",
            ),
        ):
            for i0, i1, i2, i3, i4, i5 in T.grid(
                T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)
            ):
                with Ts.sblock("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1 = Ts.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[i0_1, i2_1, i4_1, i5_1])
                    Ts.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1] = rxplaceholder[i0_1, i2_1, i4_1, i5_1]

        @R.function
        def main(
            x: R.Tensor((2, 4096, 320), dtype="float32")
        ) -> R.Tensor((2, 1, 4096, 1, 5, 64), dtype="float32"):
            cls = Module
            with R.dataflow():
                y = R.call_tir(cls.reshape, (x,), ty_args=[R.Tensor((2, 4096, 5, 64), dtype="float32")])
                z = R.call_tir(
                    cls.expand_dims, (y,), ty_args=[R.Tensor((2, 1, 4096, 1, 5, 64), "float32")]
                )
                R.output(z)
            return z

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def expand_dims(rxplaceholder: T.Tensor((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32"), expand_dims_1: T.Tensor((T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)), "float32")):
            # with Ts.sblock("root"):
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)):
                with Ts.sblock("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1 = Ts.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[i0_1, i2_1, i4_1, i5_1])
                    Ts.writes(expand_dims_1[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1])
                    expand_dims_1[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1] = rxplaceholder[i0_1, i2_1, i4_1, i5_1]

        @Ts.function
        def reshape(rxplaceholder: T.Tensor((T.int64(2), T.int64(4096), T.int64(320)), "float32"), T_reshape: T.Tensor((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32")):
            # with Ts.sblock("root"):
            for ax0_ax1_ax2_ax3_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "blockIdx.x"}):
                for ax0_ax1_ax2_ax3_fused_2 in T.parallel(T.int64(1024), annotations={"thread_binding": "threadIdx.x"}):
                    for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(10)):
                        with Ts.sblock("T_reshape"):
                            v_ax0 = Ts.axis.spatial(T.int64(2), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(1310720))
                            v_ax1 = Ts.axis.spatial(T.int64(4096), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(1310720) // T.int64(320))
                            v_ax2 = Ts.axis.spatial(T.int64(5), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(320) // T.int64(64))
                            v_ax3 = Ts.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(64))
                            Ts.reads(rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)])
                            Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)]

        @R.function
        def main(x: R.Tensor((2, 4096, 320), dtype="float32")) -> R.Tensor((2, 1, 4096, 1, 5, 64), dtype="float32"):
            cls = Expected
            with R.dataflow():
                y: R.Tensor((2, 4096, 5, 64), dtype="float32") = R.reshape(x, R.shape([2, 4096, 5, 64]))
                z = R.call_tir(cls.expand_dims, (y,), ty_args=[R.Tensor((2, 1, 4096, 1, 5, 64), dtype="float32")])
                R.output(z)
            return z
    # fmt: on

    assert relax.analysis.has_reshape_pattern(Module["reshape"])
    mod = relax.transform.RewriteDataflowReshape()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_reshape_dynamic_shape():
    n = T.dynamic("n", "int32")

    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function(private=True)
        def reshape(
            A: T.Tensor((n, 16, 128), "float16"), T_reshape: T.Tensor((1, n, 16, 128), "float16")
        ):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0_ax1_ax2_fused_0 in T.parallel(
                n * 2, annotations={"thread_binding": "blockIdx.x"}
            ):
                for ax0_ax1_ax2_fused_1 in T.parallel(
                    1024, annotations={"thread_binding": "threadIdx.x"}
                ):
                    with Ts.sblock("T_reshape"):
                        v0 = Ts.axis.spatial(
                            n, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) // 2048
                        )
                        v1 = Ts.axis.spatial(
                            16, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % 2048 // 128
                        )
                        v2 = Ts.axis.spatial(
                            128, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % 128
                        )
                        Ts.reads(
                            A[((v2 // 128 + v1) // 32 + v0) % n, (v2 // 128 + v1) % 32, v2 % 128]
                        )
                        Ts.writes(T_reshape[0, v0, v1, v2])
                        T_reshape[0, v0, v1, v2] = A[
                            ((v2 // 128 + v1) // 32 + v0) % n, (v2 // 128 + v1) % 32, v2 % 128
                        ]

        @R.function
        def main(x: R.Tensor((8, 16, 128), dtype="float16")) -> R.Tensor(
            (1, 8, 16, 128), dtype="float16"
        ):
            cls = Module
            with R.dataflow():
                y = R.call_tir(
                    cls.reshape, (x,), ty_args=[R.Tensor((1, 8, 16, 128), dtype="float16")]
                )
                z = R.add(y, R.const(1, "float16"))
                R.output(z)
            return z

    n = T.dynamic("n", "int32")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def reshape(
            A: T.Tensor((n, 16, 128), "float16"), T_reshape: T.Tensor((1, n, 16, 128), "float16")
        ):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0_ax1_ax2_fused_0 in T.parallel(
                n * 2, annotations={"thread_binding": "blockIdx.x"}
            ):
                for ax0_ax1_ax2_fused_1 in T.parallel(
                    1024, annotations={"thread_binding": "threadIdx.x"}
                ):
                    with Ts.sblock("T_reshape"):
                        v0 = Ts.axis.spatial(
                            n, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) // 2048
                        )
                        v1 = Ts.axis.spatial(
                            16, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % 2048 // 128
                        )
                        v2 = Ts.axis.spatial(
                            128, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % 128
                        )
                        Ts.reads(
                            A[((v2 // 128 + v1) // 32 + v0) % n, (v2 // 128 + v1) % 32, v2 % 128]
                        )
                        Ts.writes(T_reshape[0, v0, v1, v2])
                        T_reshape[0, v0, v1, v2] = A[
                            ((v2 // 128 + v1) // 32 + v0) % n, (v2 // 128 + v1) % 32, v2 % 128
                        ]

        @R.function
        def main(x: R.Tensor((8, 16, 128), dtype="float16")) -> R.Tensor(
            (1, 8, 16, 128), dtype="float16"
        ):
            with R.dataflow():
                y: R.Tensor((1, 8, 16, 128), dtype="float16") = R.reshape(
                    x, R.shape([1, 8, 16, 128])
                )
                z: R.Tensor((1, 8, 16, 128), dtype="float16") = R.add(y, R.const(1, "float16"))
                R.output(z)
            return z

    assert relax.analysis.has_reshape_pattern(Module["reshape"])
    mod = relax.transform.RewriteDataflowReshape()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_reshape_non_dataflow():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def reshape(
            rxplaceholder: T.Tensor((T.int64(8), T.int64(3)), "float32"),
            T_reshape: T.Tensor((T.int64(2), T.int64(4), T.int64(3)), "float32"),
        ):
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(3)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(
                        rxplaceholder[
                            (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                            (v_ax0 * 12 + v_ax1 * 3 + v_ax2) % T.int64(3),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) % T.int64(3),
                    ]

        @R.function
        def main(x: R.Tensor((8, 3), dtype="float32")) -> R.Tensor((2, 4, 3), dtype="float32"):
            cls = Module
            y = R.call_tir(cls.reshape, (x,), ty_args=[R.Tensor((2, 4, 3), dtype="float32")])
            return y

    assert relax.analysis.has_reshape_pattern(Module["reshape"])
    # The binding var of the call_tir is not a DataflowVar. So the pass does no change.
    mod = relax.transform.RewriteDataflowReshape()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Module)


def test_tuple_get_reshape():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def fused_reshape5(
            lv2_0: T.Tensor((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            lv2_1: T.Tensor((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            lv2_2: T.Tensor((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            T_reshape_handle_intermediate: T.Tensor(
                (T.int64(2), T.int64(4096), T.int64(8), T.int64(40)), "float16"
            ),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4096), T.int64(8), T.int64(40)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(
                        lv2_0[
                            (
                                ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1)
                                // T.int64(4096)
                                + v_ax0
                            )
                            % T.int64(2),
                            ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096),
                            (v_ax2 * T.int64(40) + v_ax3) % T.int64(320),
                        ]
                    )
                    Ts.writes(T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv2_0[
                        (
                            ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096)
                            + v_ax0
                        )
                        % T.int64(2),
                        ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096),
                        (v_ax2 * T.int64(40) + v_ax3) % T.int64(320),
                    ]

        @R.function
        def main(
            lv41_1: R.Tuple(
                R.Tensor((2, 4096, 320), dtype="float16"),
                R.Tensor((2, 4096, 320), dtype="float16"),
                R.Tensor((2, 4096, 320), dtype="float16"),
            ),
        ) -> R.Tensor((2, 4096, 8, 40), dtype="float16"):
            cls = Module
            with R.dataflow():
                lv: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[0]
                lv1: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[1]
                lv2: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[2]
                lv645 = R.call_tir(
                    cls.fused_reshape5,
                    (lv, lv1, lv2),
                    ty_args=[R.Tensor((2, 4096, 8, 40), dtype="float16")],
                )
                out: R.Tensor((2, 4096, 8, 40), dtype="float16") = R.add(lv645, lv645)
                R.output(out)
            return out

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def fused_reshape5(
            lv2_0: T.Tensor((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            lv2_1: T.Tensor((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            lv2_2: T.Tensor((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            T_reshape_handle_intermediate: T.Tensor(
                (T.int64(2), T.int64(4096), T.int64(8), T.int64(40)), "float16"
            ),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4096), T.int64(8), T.int64(40)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(
                        lv2_0[
                            (
                                ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1)
                                // T.int64(4096)
                                + v_ax0
                            )
                            % T.int64(2),
                            ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096),
                            (v_ax2 * T.int64(40) + v_ax3) % T.int64(320),
                        ]
                    )
                    Ts.writes(T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv2_0[
                        (
                            ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096)
                            + v_ax0
                        )
                        % T.int64(2),
                        ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096),
                        (v_ax2 * T.int64(40) + v_ax3) % T.int64(320),
                    ]

        @R.function
        def main(
            lv41_1: R.Tuple(
                R.Tensor((2, 4096, 320), dtype="float16"),
                R.Tensor((2, 4096, 320), dtype="float16"),
                R.Tensor((2, 4096, 320), dtype="float16"),
            ),
        ) -> R.Tensor((2, 4096, 8, 40), dtype="float16"):
            with R.dataflow():
                lv: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[0]
                lv1: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[1]
                lv2: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[2]
                lv645: R.Tensor((2, 4096, 8, 40), dtype="float16") = R.reshape(
                    lv, R.shape([2, 4096, 8, 40])
                )
                out: R.Tensor((2, 4096, 8, 40), dtype="float16") = R.add(lv645, lv645)
                R.output(out)
            return out

    rewritten = relax.transform.RewriteDataflowReshape()(Module)
    tvm_tirx.ir.assert_structural_equal(rewritten, Expected)


def test_invalid_reshape():
    @tvm_tirx.script.ir_module
    class Module:
        # The strided_slice op has the reshape pattern, but it can take only a part of the input.
        # It can't be replaced with the reshape op because reshape expects to preserve the "volume"
        # of the input.
        @Ts.function
        def strided_slice(
            A: T.Tensor((T.int64(1), T.int64(1024)), "int32"),
            T_strided_slice: T.Tensor((T.int64(1), T.int64(1000)), "int32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1000)):
                with Ts.sblock("T_strided_slice"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_strided_slice[v_ax0, v_ax1])
                    T_strided_slice[v_ax0, v_ax1] = A[v_ax0, v_ax1]

        @Ts.function
        def add_one(
            A: T.Tensor((T.int64(1), T.int64(1000)), "int32"),
            T_add_one: T.Tensor((T.int64(1), T.int64(1000)), "int32"),
        ):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1000)):
                with Ts.sblock("T_add_one"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_add_one[v_ax0, v_ax1])
                    T_add_one[v_ax0, v_ax1] = A[v_ax0, v_ax1] + 1

        @R.function
        def main(A: R.Tensor((1, 1024), dtype="int32")) -> R.Tensor((1, 1000), dtype="int32"):
            with R.dataflow():
                cls = Module
                S = R.call_tir(
                    cls.strided_slice, (A,), ty_args=[R.Tensor((1, 1000), dtype="int32")]
                )
                A = R.call_tir(cls.add_one, (S,), ty_args=[R.Tensor((1, 1000), dtype="int32")])
                R.output(A)
            return A

    assert relax.analysis.has_reshape_pattern(Module["strided_slice"])
    rewritten = relax.transform.RewriteDataflowReshape()(Module)
    tvm_tirx.ir.assert_structural_equal(rewritten, Module)


def test_reshape_detect_nop():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((8, 8), dtype="float16")) -> R.Tensor((8, 8), dtype="float16"):
            with R.dataflow():
                gv = R.call_pure_packed("foo", x, x, ty_args=(R.Tensor((8, 8), dtype="float16"),))
                out = R.call_pure_packed(
                    "foo", gv, gv, ty_args=(R.Tensor((8, 8), dtype="float16"),)
                )
                R.output(out)
            return out

    rewritten = relax.transform.RewriteDataflowReshape()(Module)
    tvm_tirx.ir.assert_structural_equal(rewritten, Module)


def test_reshape_scalar():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((1,), dtype="float32") = R.reshape(x, [1])
                lv2: R.Tensor((1,), dtype="float32") = R.add(lv1, lv1)
                R.output(lv2)
            return lv2

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def add(
            A: T.Tensor((T.int64(1),), "float32"),
            B: T.Tensor((T.int64(1),), "float32"),
            T_add: T.Tensor((T.int64(1),), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0 in range(T.int64(1)):
                with Ts.sblock("T_add"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                    Ts.reads(A[v_ax0], B[v_ax0])
                    Ts.writes(T_add[v_ax0])
                    T_add[v_ax0] = A[v_ax0] + B[v_ax0]

        @Ts.function(private=True)
        def reshape(A: T.Tensor((), "float32"), T_reshape: T.Tensor((T.int64(1),), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0 in range(T.int64(1)):
                with Ts.sblock("T_reshape"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                    Ts.reads(A[()])
                    Ts.writes(T_reshape[v_ax0])
                    T_reshape[v_ax0] = A[()]

        @R.function
        def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv1: R.Tensor((1,), dtype="float32") = R.reshape(x, R.shape([1]))
                lv2 = R.call_tir(cls.add, (lv1, lv1), ty_args=[R.Tensor((1,), dtype="float32")])
                R.output(lv2)
            return lv2

    mod = Module
    mod = relax.transform.LegalizeOps()(mod)
    rewritten = relax.transform.RewriteDataflowReshape()(mod)
    tvm_tirx.ir.assert_structural_equal(rewritten, Expected)


def test_rewrite_static_reshape():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor([256], dtype="float32")):
            with R.dataflow():
                y = R.reshape(x, [64, 4])
                z = R.add(y, y)
                R.output(z)
            return z

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((256,), dtype="float32")):
            cls = Expected

            with R.dataflow():
                y = R.reshape(x, R.shape([64, 4]))
                z = R.call_tir(cls.add, (y, y), ty_args=[R.Tensor((64, 4), dtype="float32")])
                R.output(z)
            return z

        @Ts.function(private=True)
        def add(
            y1: T.Tensor((T.int64(64), T.int64(4)), "float32"),
            y2: T.Tensor((T.int64(64), T.int64(4)), "float32"),
            z: T.Tensor((T.int64(64), T.int64(4)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            for (*iters,) in T.grid(T.int64(64), T.int64(4)):
                with Ts.sblock("T_add"):
                    i, j = Ts.axis.remap("SS", iters)
                    z[i, j] = y1[i, j] + y2[i, j]

    After = tvm_tirx.ir.transform.Sequential(
        [
            # Lower both R.reshape and R.add from Relax to TIR
            relax.transform.LegalizeOps(),
            # Identify reshapes, raise calls to cls.reshape from TIR
            # to Relax
            relax.transform.RewriteDataflowReshape(),
            # Clean up afterwards, removing the no-longer-required
            # Function "reshape"
            relax.transform.DeadCodeElimination(),
        ]
    )(Before)

    tvm_tirx.ir.assert_structural_equal(Expected, After)


# def test_rewrite_dynamic_reshape():
#     N = T.dynamic("N", "int64")
#
#     @I.ir_module
#     class Before:
#         @R.function
#         def main(x: R.Tensor([N], dtype="float32")):
#             with R.dataflow():
#                 y = R.reshape(x, [N // 4, 4])
#                 z = R.add(y, y)
#                 R.output(z)
#             return z

#     @I.ir_module
#     class Expected:
#         @R.function
#         def main(x: R.Tensor([N], dtype="float32")):
#             cls = Expected

#             with R.dataflow():
#                 y = R.reshape(x, R.shape([N // 4, 4]))
#                 z = R.call_tir(
#                     cls.add,
#                     (y, y, N),
#                     out_ty=R.Tensor((N // 4, 4), dtype="float32"),
#                 )
#                 R.output(z)
#             return z

#         @T.function(private=True)
#         def add(
#             y1: T.Tensor([N // 4, 4], "float32"),
#             y2: T.Tensor([N // 4, 4], "float32"),
#             N: T.int64,
#             z: T.Tensor([N // 4, 4], "float32"),
#         ):


#             T.func_attr({"tirx.noalias": True})

#             for iters in T.grid(T.int64(64), T.int64(4)):
#                 with Ts.sblock("T_add"):
#                     i, j = Ts.axis.remap("SS", iters)
#                     z[i, j] = y1[i, j] + y2[i, j]

#     After = tvm.ir.transform.Sequential(
#         [
#             # Lower both R.reshape and R.add from Relax to TIR
#             relax.transform.LegalizeOps(),
#             # Identify reshapes, raise calls to cls.reshape from TIR
#             # to Relax
#             relax.transform.RewriteDataflowReshape(),
#             # Clean up afterwards, removing the no-longer-required
#             # Function "reshape"
#             relax.transform.DeadCodeElimination(),
#         ]
#     )(Before)
#     After.show()
#     tvm.ir.assert_structural_equal(Expected, After)


def test_rewrite_dynamic_reshape():
    N = T.dynamic("N", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor([N, 16], dtype="float32")):
            with R.dataflow():
                y = R.reshape(x, [N * 4, T.int64(4)])
                z = R.add(y, y)
                R.output(z)
            return z

    N = T.dynamic("N", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor([N, 16], dtype="float32")):
            cls = Expected

            with R.dataflow():
                y = R.reshape(x, R.shape([N * 4, T.int64(4)]))
                z = R.call_tir(
                    cls.add,
                    (y, y, N),
                    ty_args=[R.Tensor((N * 4, 4), dtype="float32")],
                )
                R.output(z)
            return z

        add_N = T.int64()

        @Ts.function(private=True)
        def add(
            y1: T.Tensor([add_N * 4, T.int64(4)], "float32"),
            y2: T.Tensor([add_N * 4, T.int64(4)], "float32"),
            N: add_N,
            z: T.Tensor([add_N * 4, T.int64(4)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            for (*iters,) in T.grid(N * 4, T.int64(4)):
                with Ts.sblock("T_add"):
                    i, j = Ts.axis.remap("SS", iters)
                    z[i, j] = y1[i, j] + y2[i, j]

    After = tvm_tirx.ir.transform.Sequential(
        [
            # Lower both R.reshape and R.add from Relax to TIR
            relax.transform.LegalizeOps(),
            # Identify reshapes, raise calls to cls.reshape from TIR
            # to Relax
            relax.transform.RewriteDataflowReshape(),
            # Clean up afterwards, removing the no-longer-required
            # Function "reshape"
            relax.transform.DeadCodeElimination(),
        ]
    )(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
