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
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.frontend import detach_params
from tvm.relax.frontend.common import autopad
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.script.parser import relax as R


def test_detach_params():
    @R.function
    def func(x: R.Tensor((2, 3), "float32")):
        return x

    param = tvm_tirx.runtime.empty((3,), "float32")
    mod = tvm_tirx.IRModule({"func": func.with_attr("params", [param])})
    detached_mod, detached_params = detach_params(mod)

    tvm_tirx.ir.assert_structural_equal(detached_mod, tvm_tirx.IRModule({"func": func}))
    assert len(detached_params) == 1
    assert "func" in detached_params
    assert isinstance(detached_params["func"], list)
    assert len(detached_params["func"]) == 1
    tvm_tirx.testing.assert_allclose(detached_params["func"][0].numpy(), param.numpy())


class TestAutopad:
    def _test_autopad(self, pad_type, expected):
        bb = relax.BlockBuilder()
        input_shape = (1, 1, 4, 4)
        x = relax.Var("x", relax.TensorType(input_shape, "float32"))

        with bb.function("main", [x]):
            with bb.dataflow():
                result = autopad(
                    bb,
                    x,
                    strides=[2, 2],
                    kernel_shape=[3, 3],
                    dilations=(1, 1),
                    pad_type=pad_type,
                    deconv=False,
                    mode="SAME_UPPER",
                    pad_value=0.0,
                )
                out = bb.emit_output(result)
            bb.emit_func_output(out)

        tvm_tirx.ir.assert_structural_equal(bb.get(), expected)

    def test_constant(self):
        @I.ir_module
        class expected:
            @Ts.function(private=True)
            def pad(
                x: T.Tensor((T.int64(1), T.int64(1), T.int64(4), T.int64(4)), "float32"),
                PadInput: T.Tensor((T.int64(1), T.int64(1), T.int64(5), T.int64(5)), "float32"),
            ):
                T.func_attr({"tirx.noalias": True})
                for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(5), T.int64(5)):
                    with Ts.sblock("PadInput"):
                        v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                        Ts.reads(x[v_i0, v_i1, v_i2, v_i3])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                        PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                            T.int64(0) <= v_i2
                            and v_i2 < T.int64(4)
                            and T.int64(0) <= v_i3
                            and v_i3 < T.int64(4),
                            x[v_i0, v_i1, v_i2, v_i3],
                            T.float32(0.0),
                        )

            @R.function
            def main(x: R.Tensor((1, 1, 4, 4), dtype="float32")) -> R.Tensor(
                (1, 1, 5, 5), dtype="float32"
            ):
                cls = expected
                with R.dataflow():
                    lv = R.call_tir(
                        cls.pad, (x,), ty_args=[R.Tensor((1, 1, 5, 5), dtype="float32")]
                    )
                    gv: R.Tensor((1, 1, 5, 5), dtype="float32") = lv
                    R.output(gv)
                return gv

        self._test_autopad("constant", expected)

    def test_edge(self):
        @I.ir_module
        class expected:
            @Ts.function(private=True)
            def replicate_pad(
                x: T.Tensor((T.int64(1), T.int64(1), T.int64(4), T.int64(4)), "float32"),
                ReplicatePadInput: T.Tensor(
                    (T.int64(1), T.int64(1), T.int64(5), T.int64(5)), "float32"
                ),
            ):
                T.func_attr({"tirx.noalias": True})
                for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(5), T.int64(5)):
                    with Ts.sblock("ReplicatePadInput"):
                        v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                        Ts.reads(
                            x[
                                T.int64(0),
                                T.int64(0),
                                T.max(T.int64(0), T.min(T.int64(3), v_i2)),
                                T.max(T.int64(0), T.min(T.int64(3), v_i3)),
                            ]
                        )
                        Ts.writes(ReplicatePadInput[v_i0, v_i1, v_i2, v_i3])
                        ReplicatePadInput[v_i0, v_i1, v_i2, v_i3] = x[
                            T.int64(0),
                            T.int64(0),
                            T.max(T.int64(0), T.min(T.int64(3), v_i2)),
                            T.max(T.int64(0), T.min(T.int64(3), v_i3)),
                        ]

            @R.function
            def main(x: R.Tensor((1, 1, 4, 4), dtype="float32")) -> R.Tensor(
                (1, 1, 5, 5), dtype="float32"
            ):
                cls = expected
                with R.dataflow():
                    lv = R.call_tir(
                        cls.replicate_pad, (x,), ty_args=[R.Tensor((1, 1, 5, 5), dtype="float32")]
                    )
                    gv: R.Tensor((1, 1, 5, 5), dtype="float32") = lv
                    R.output(gv)
                return gv

        self._test_autopad("edge", expected)

    def test_reflect(self):
        @I.ir_module
        class expected:
            @Ts.function(private=True)
            def mirror_pad(
                x: T.Tensor((T.int64(1), T.int64(1), T.int64(4), T.int64(4)), "float32"),
                MirrorPadInput: T.Tensor(
                    (T.int64(1), T.int64(1), T.int64(5), T.int64(5)), "float32"
                ),
            ):
                T.func_attr({"tirx.noalias": True})
                for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(5), T.int64(5)):
                    with Ts.sblock("MirrorPadInput"):
                        v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                        Ts.reads(x[v_i0, v_i1, T.int64(0) : T.int64(4), T.int64(0) : T.int64(4)])
                        Ts.writes(MirrorPadInput[v_i0, v_i1, v_i2, v_i3])
                        MirrorPadInput[v_i0, v_i1, v_i2, v_i3] = x[
                            v_i0,
                            v_i1,
                            T.if_then_else(
                                T.int64(4) <= v_i2,
                                T.int64(6) - v_i2,
                                T.if_then_else(v_i2 < T.int64(0), v_i2 * T.int64(-1), v_i2),
                            ),
                            T.if_then_else(
                                T.int64(4) <= v_i3,
                                T.int64(6) - v_i3,
                                T.if_then_else(v_i3 < T.int64(0), v_i3 * T.int64(-1), v_i3),
                            ),
                        ]

            @R.function
            def main(x: R.Tensor((1, 1, 4, 4), dtype="float32")) -> R.Tensor(
                (1, 1, 5, 5), dtype="float32"
            ):
                cls = expected
                with R.dataflow():
                    lv = R.call_tir(
                        cls.mirror_pad, (x,), ty_args=[R.Tensor((1, 1, 5, 5), dtype="float32")]
                    )
                    gv: R.Tensor((1, 1, 5, 5), dtype="float32") = lv
                    R.output(gv)
                return gv

        self._test_autopad("reflect", expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
