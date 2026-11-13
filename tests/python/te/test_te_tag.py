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
# ruff: noqa: E741, F401, F841
import json

import tvm
import tvm_tirx
from tvm import te


@tvm.te.tag_scope(tag="conv")
def compute_conv(data, weight):
    N, IC, H, W = data.shape
    OC, IC, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    ic = te.reduce_axis((0, IC), name="ic")
    dh = te.reduce_axis((0, KH), name="dh")
    dw = te.reduce_axis((0, KW), name="dw")

    return te.compute(
        (N, OC, OH, OW),
        lambda i, oc, h, w: te.sum(
            data[i, ic, h + dh, w + dw] * weight[oc, ic, dh, dw], axis=[ic, dh, dw]
        ),
    )


def test_with():
    n = te.var("n")
    m = te.var("m")
    l = te.var("l")

    A = te.placeholder((n, l), name="A")
    B = te.placeholder((m, l), name="B")
    with tvm.te.tag_scope(tag="gemm"):
        k = te.reduce_axis((0, l), name="k")
        C = te.compute(
            (n, m),
            lambda i, j: te.sum(A[i, k] * B[j, k], axis=k),
            attrs={"hello": 1, "arr": [10, 12]},
        )

    assert C.op.tag == "gemm"
    assert "hello" in C.op.attrs
    assert "xx" not in C.op.attrs
    assert C.op.attrs["hello"] == 1
    CC = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(C))
    assert CC.op.attrs["hello"] == 1
    assert len(CC.op.attrs["arr"]) == 2
    assert CC.op.attrs["arr"][0] == 10
    assert CC.op.attrs["arr"][1] == 12


def test_decorator():
    n = te.var("n")
    c = te.var("c")
    h = te.var("h")
    w = te.var("w")
    kh = te.var("kh")
    kw = te.var("kw")

    A = te.placeholder((n, c, h, w), name="A")
    B = te.placeholder((c, c, kh, kw), name="B")
    C = compute_conv(A, B)
    assert C.op.tag == "conv"
    assert len(C.op.attrs) == 0


def test_nested():
    n = te.var("n")
    c = te.var("c")
    h = te.var("h")
    w = te.var("w")
    kh = te.var("kh")
    kw = te.var("kw")

    A = te.placeholder((n, c, h, w), name="A")
    B = te.placeholder((c, c, kh, kw), name="B")
    try:
        with te.tag_scope(tag="conv"):
            C = compute_conv(A, B)
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    test_with()
    test_decorator()
    test_nested()
