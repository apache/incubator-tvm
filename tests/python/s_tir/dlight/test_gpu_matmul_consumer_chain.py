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
"""Matmul must decline consumer chains it cannot inline (issue #20615)."""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm.s_tir.dlight.gpu.matmul import auto_inline_consumer_chain
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


@pytest.fixture(params=[False, True], ids=["bias_relu", "bias_relu_pad"])
def conv_consumer_chain(request):
    pad = int(request.param)

    @Ts.function
    def before(
        data: T.Tensor((1, 8, 9, 17), "float32"),
        weight: T.Tensor((4, 8, 2, 2), "float32"),
        bias: T.Tensor((1, 4, 1, 1), "float32"),
        output: T.Tensor((1, 4, 8 + pad, 16 + pad), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        padded_input = Ts.sblock_alloc_tensor((1, 8, 9, 17))
        conv = Ts.sblock_alloc_tensor((1, 4, 8, 16))
        add = Ts.sblock_alloc_tensor((1, 4, 8, 16))
        relu = Ts.sblock_alloc_tensor((1, 4, 8, 16))

        for n, c, h, w in T.grid(1, 8, 9, 17):
            with Ts.sblock("copy_input"):
                vn, vc, vh, vw = Ts.axis.remap("SSSS", [n, c, h, w])
                padded_input[vn, vc, vh, vw] = data[vn, vc, vh, vw]

        for n, f, y, x, rc, ry, rx in T.grid(1, 4, 8, 16, 8, 2, 2):
            with Ts.sblock("conv2d_nchw"):
                vn, vf, vy, vx, vrc, vry, vrx = Ts.axis.remap("SSSSRRR", [n, f, y, x, rc, ry, rx])
                with Ts.init():
                    conv[vn, vf, vy, vx] = T.float32(0)
                conv[vn, vf, vy, vx] += (
                    padded_input[vn, vrc, vy + vry, vx + vrx] * weight[vf, vrc, vry, vrx]
                )

        for n, f, y, x in T.grid(1, 4, 8, 16):
            with Ts.sblock("bias_add"):
                vn, vf, vy, vx = Ts.axis.remap("SSSS", [n, f, y, x])
                add[vn, vf, vy, vx] = conv[vn, vf, vy, vx] + bias[vn, vf, 0, 0]

        for n, f, y, x in T.grid(1, 4, 8, 16):
            with Ts.sblock("relu"):
                vn, vf, vy, vx = Ts.axis.remap("SSSS", [n, f, y, x])
                relu[vn, vf, vy, vx] = T.max(add[vn, vf, vy, vx], T.float32(0))

        for n, f, y, x in T.grid(1, 4, 8 + pad, 16 + pad):
            with Ts.sblock("pad_output"):
                vn, vf, vy, vx = Ts.axis.remap("SSSS", [n, f, y, x])
                output[vn, vf, vy, vx] = T.if_then_else(
                    vy < 8 and vx < 16, relu[vn, vf, vy, vx], T.float32(0)
                )

    return before, bool(pad)


def test_matmul_consumer_chain(conv_consumer_chain):
    before, padded = conv_consumer_chain
    original = tvm_tirx.IRModule({"main": before})
    snapshot = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(original))
    target = Target("cuda")
    schedule = dl.gpu.Matmul().apply(before, target, False)
    assert (schedule is None) == padded
    assert_structural_equal(original, snapshot)

    with target:
        matmul_only = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(original)
        if padded:
            assert_structural_equal(matmul_only, snapshot)
        else:
            assert matmul_only["main"].attrs["tirx.is_scheduled"] == 1


def test_matmul_consumer_chain_following_rules(conv_consumer_chain):
    before, padded = conv_consumer_chain
    following_rules = [
        dl.gpu.GEMV(),
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Fallback(),
    ]
    with Target("cuda"):
        actual = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), *following_rules)(
            tvm_tirx.IRModule({"main": before})
        )
        if padded:
            expected = dl.ApplyDefaultSchedule(*following_rules)(
                tvm_tirx.IRModule({"main": before})
            )
            assert_structural_equal(actual, expected)
    assert actual["main"].attrs["tirx.is_scheduled"] == 1


def test_consumer_chain_non_inlineable_producer():
    @Ts.function
    def before(
        a: T.Tensor((16, 16), "float32"),
        b: T.Tensor((16, 16), "float32"),
        output: T.Tensor((16,), "float32"),
    ):
        lhs = Ts.sblock_alloc_tensor((16,))
        rhs = Ts.sblock_alloc_tensor((16,))
        for i, k in T.grid(16, 16):
            with Ts.sblock("lhs"):
                vi, vk = Ts.axis.remap("SR", [i, k])
                with Ts.init():
                    lhs[vi] = T.float32(0)
                lhs[vi] += a[vi, vk]
        for i, k in T.grid(16, 16):
            with Ts.sblock("rhs"):
                vi, vk = Ts.axis.remap("SR", [i, k])
                with Ts.init():
                    rhs[vi] = T.float32(0)
                rhs[vi] += b[vi, vk]
        for i in range(16):
            with Ts.sblock("add"):
                vi = Ts.axis.spatial(16, i)
                output[vi] = lhs[vi] + rhs[vi]

    sch = s_tir.Schedule(before)
    assert not auto_inline_consumer_chain(sch, sch.get_sblock("lhs"))


if __name__ == "__main__":
    tvm_tirx.testing.main()
