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

import numpy as np

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.transform import ToMixedPrecision
from tvm.s_tir import script as Ts
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def _assert_test(input, expected=None, expected2=None):
    if expected:
        mod = ToMixedPrecision()(input)
        tvm_tirx.ir.assert_structural_equal(mod, expected)

    if expected2:
        mod = ToMixedPrecision(out_dtype="float16")(input)
        tvm_tirx.ir.assert_structural_equal(mod, expected2)


def test_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv2: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.astype(lv2, dtype="float32")
                R.output(gv)
            return gv

    _assert_test(Input, Expected, Expected2)


def test_conv2d_relu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(lv)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv_1: R.Tensor((2, 4, 26, 26), dtype="float16") = R.astype(lv2, dtype="float16")
                lv3: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.relu(lv_1)
                gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.astype(lv3, dtype="float32")
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv_1: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.relu(lv_1)
                gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.astype(lv2, dtype="float32")
                R.output(gv)
            return gv

    _assert_test(Input, Expected, Expected2)


def test_unknown_dtype_is_not_rewritten():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 3), dtype=None)) -> R.Tensor((2, 3), dtype=None):
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype=None) = R.nn.relu(x)
                R.output(gv)
            return gv

    mod = ToMixedPrecision()(Input)
    tvm_tirx.ir.assert_structural_equal(mod, Input)


def test_relu_conv2d_relu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                x0: R.Tensor((2, 3, 28, 28), "float32") = R.nn.relu(x)
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x0, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                x0: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.relu(x)
                lv1: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x0, dtype="float16")
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                    lv1,
                    lv,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 4, 26, 26), dtype="float16") = R.astype(lv2, dtype="float16")
                lv3: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.relu(gv)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.astype(lv3, dtype="float32")
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                x0: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.relu(x)
                lv1: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x0, dtype="float16")
                gv: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.conv2d(
                    lv1,
                    lv,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.relu(gv)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.astype(lv2, dtype="float32")
                R.output(gv2)
            return gv2

    _assert_test(Input, Expected, Expected2)


def test_conv2d_relu_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            w2: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 4, 24, 24), "float32") = R.nn.conv2d(gv2, w2, out_dtype="float32")
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            w2: R.Tensor((4, 4, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv2: R.Tensor((4, 4, 3, 3), dtype="float16") = R.astype(w2, dtype="float16")
                lv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 4, 26, 26), dtype="float16") = R.astype(lv3, dtype="float16")
                gv2: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.relu(gv)
                gv3: R.Tensor((2, 4, 24, 24), dtype="float32") = R.nn.conv2d(
                    gv2,
                    lv2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            w2: R.Tensor((4, 4, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv2: R.Tensor((4, 4, 3, 3), dtype="float16") = R.astype(w2, dtype="float16")
                gv: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.relu(gv)
                lv3: R.Tensor((2, 4, 24, 24), dtype="float16") = R.nn.conv2d(
                    gv2,
                    lv2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                gv3: R.Tensor((2, 4, 24, 24), dtype="float32") = R.astype(lv3, dtype="float32")
                R.output(gv3)
            return gv3

    _assert_test(Input, Expected, Expected2)


def test_gemm_add_silu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 320), "float32"),
            w1: R.Tensor((320, 1280), "float32"),
            w2: R.Tensor((2, 1280), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv0: R.Tensor((2, 1280), "float32") = R.matmul(x, w1, out_dtype="float32")
                gv1: R.Tensor((2, 1280), "float32") = R.add(gv0, w2)
                gv2: R.Tensor((2, 1280), "float32") = R.nn.silu(gv1)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 320), dtype="float32"),
            w1: R.Tensor((320, 1280), dtype="float32"),
            w2: R.Tensor((2, 1280), dtype="float32"),
        ) -> R.Tensor((2, 1280), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 320), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((320, 1280), dtype="float16") = R.astype(w1, dtype="float16")
                lv2: R.Tensor((2, 1280), dtype="float32") = R.matmul(lv, lv1, out_dtype="float32")
                gv0: R.Tensor((2, 1280), dtype="float16") = R.astype(lv2, dtype="float16")
                lv3: R.Tensor((2, 1280), dtype="float32") = R.astype(gv0, dtype="float32")
                gv1: R.Tensor((2, 1280), dtype="float32") = R.add(lv3, w2)
                gv2: R.Tensor((2, 1280), dtype="float32") = R.nn.silu(gv1)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((2, 320), dtype="float32"),
            w1: R.Tensor((320, 1280), dtype="float32"),
            w2: R.Tensor((2, 1280), dtype="float32"),
        ) -> R.Tensor((2, 1280), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 320), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((320, 1280), dtype="float16") = R.astype(w1, dtype="float16")
                gv0: R.Tensor((2, 1280), dtype="float16") = R.matmul(lv, lv1, out_dtype="float16")
                lv2: R.Tensor((2, 1280), dtype="float32") = R.astype(gv0, dtype="float32")
                gv1: R.Tensor((2, 1280), dtype="float32") = R.add(lv2, w2)
                gv2: R.Tensor((2, 1280), dtype="float32") = R.nn.silu(gv1)
                R.output(gv2)
            return gv2

    _assert_test(Input, Expected, Expected2)


def test_tuple():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            w_2: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv3 = (gv, gv2)
                gv4 = (gv3, gv2)
                gv5 = gv4[0]
                gv6 = gv5[0]
                gv7 = R.nn.conv2d(gv6, w_2, out_dtype="float32")
                R.output(gv7)
            return gv7

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            w_2: R.Tensor((4, 4, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv2: R.Tensor((4, 4, 3, 3), dtype="float16") = R.astype(w_2, dtype="float16")
                lv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 4, 26, 26), dtype="float16") = R.astype(lv3, dtype="float16")
                lv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float16") = R.astype(lv4, dtype="float16")
                gv3: R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                ) = (gv, gv2)
                gv4: R.Tuple(
                    R.Tuple(
                        R.Tensor((2, 4, 26, 26), dtype="float16"),
                        R.Tensor((2, 4, 26, 26), dtype="float16"),
                    ),
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                ) = (gv3, gv2)
                gv5: R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                ) = gv4[0]
                gv6: R.Tensor((2, 4, 26, 26), dtype="float16") = gv5[0]
                gv7: R.Tensor((2, 4, 24, 24), dtype="float32") = R.nn.conv2d(
                    gv6,
                    lv2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                R.output(gv7)
            return gv7

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            w_2: R.Tensor((4, 4, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv2: R.Tensor((4, 4, 3, 3), dtype="float16") = R.astype(w_2, dtype="float16")
                gv: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                gv3: R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                ) = (gv, gv2)
                gv4: R.Tuple(
                    R.Tuple(
                        R.Tensor((2, 4, 26, 26), dtype="float16"),
                        R.Tensor((2, 4, 26, 26), dtype="float16"),
                    ),
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                ) = (gv3, gv2)
                gv5: R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                    R.Tensor((2, 4, 26, 26), dtype="float16"),
                ) = gv4[0]
                gv6: R.Tensor((2, 4, 26, 26), dtype="float16") = gv5[0]
                lv3: R.Tensor((2, 4, 24, 24), dtype="float16") = R.nn.conv2d(
                    gv6,
                    lv2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                gv7: R.Tensor((2, 4, 24, 24), dtype="float32") = R.astype(lv3, dtype="float32")
                R.output(gv7)
            return gv7

    _assert_test(Input, Expected, Expected2)


def test_concat_matmul():
    @I.ir_module
    class Input:
        @R.function
        def main(
            lv10: R.Tensor((2, 160), "float32"),
            lv12: R.Tensor((2, 160), "float32"),
            w: R.Tensor((320, 1280), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                lv13: R.Tensor((2, 320), "float32") = R.concat((lv10, lv12), axis=-1)
                lv14: R.Tensor((2, 1280), "float32") = R.matmul(lv13, w, out_dtype="float32")
                R.output(lv14)
            return lv14

    @I.ir_module
    class Expected:
        @R.function
        def main(
            lv10: R.Tensor((2, 160), dtype="float32"),
            lv12: R.Tensor((2, 160), dtype="float32"),
            w: R.Tensor((320, 1280), dtype="float32"),
        ) -> R.Tensor((2, 1280), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((320, 1280), dtype="float16") = R.astype(w, dtype="float16")
                lv13: R.Tensor((2, 320), dtype="float32") = R.concat((lv10, lv12), axis=-1)
                lv1: R.Tensor((2, 320), dtype="float16") = R.astype(lv13, dtype="float16")
                lv14: R.Tensor((2, 1280), dtype="float32") = R.matmul(lv1, lv, out_dtype="float32")
                R.output(lv14)
            return lv14

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            lv10: R.Tensor((2, 160), dtype="float32"),
            lv12: R.Tensor((2, 160), dtype="float32"),
            w: R.Tensor((320, 1280), dtype="float32"),
        ) -> R.Tensor((2, 1280), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((320, 1280), dtype="float16") = R.astype(w, dtype="float16")
                lv13: R.Tensor((2, 320), dtype="float32") = R.concat((lv10, lv12), axis=-1)
                lv1: R.Tensor((2, 320), dtype="float16") = R.astype(lv13, dtype="float16")
                lv2: R.Tensor((2, 1280), dtype="float16") = R.matmul(lv1, lv, out_dtype="float16")
                lv14: R.Tensor((2, 1280), dtype="float32") = R.astype(lv2, dtype="float32")
                R.output(lv14)
            return lv14

    _assert_test(Input, Expected, Expected2)


def test_conv2d_softmax():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((3, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 3, 28, 28), "float32") = R.nn.conv2d(x, w, padding=(1, 1))
                gv1: R.Tensor((2, 3, 28, 28), "float32") = R.nn.softmax(x, axis=1)
                gv2 = R.add(gv, gv1)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((3, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 3, 28, 28), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv1: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv2: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.conv2d(
                    lv1,
                    lv,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(lv2, dtype="float16")
                gv1: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.softmax(x, axis=1)
                lv3: R.Tensor((2, 3, 28, 28), dtype="float32") = R.astype(gv, dtype="float32")
                gv2: R.Tensor((2, 3, 28, 28), dtype="float32") = R.add(lv3, gv1)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((3, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 3, 28, 28), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv1: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.nn.conv2d(
                    lv1,
                    lv,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                gv1: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.softmax(x, axis=1)
                lv2: R.Tensor((2, 3, 28, 28), dtype="float32") = R.astype(gv, dtype="float32")
                gv2: R.Tensor((2, 3, 28, 28), dtype="float32") = R.add(lv2, gv1)
                R.output(gv2)
            return gv2

    _assert_test(Input, Expected, Expected2)


def test_conv2d_bias_conv2d():
    @tvm_tirx.script.ir_module
    class Input:
        @R.function
        def main(
            z: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w0: R.Tensor((512, 4, 3, 3), dtype="float16"),
            w1: R.Tensor((512,), dtype="float16"),
            w2: R.Tensor((4, 4, 1, 1), dtype="float16"),
            w3: R.Tensor((4,), dtype="float16"),
        ) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((512, 4, 3, 3), dtype="float32") = R.wrap_param(w0, dtype="float32")
                lv1: R.Tensor((512,), dtype="float32") = R.wrap_param(w1, dtype="float32")
                lv140: R.Tensor((4, 4, 1, 1), dtype="float32") = R.wrap_param(w2, dtype="float32")
                lv141: R.Tensor((4,), dtype="float32") = R.wrap_param(w3, dtype="float32")
                lv142: R.Tensor((1, 4, 64, 64), dtype="float32") = R.nn.conv2d(
                    z,
                    lv140,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv143: R.Tensor((1, 4, 1, 1), dtype="float32") = R.reshape(lv141, (1, 4, 1, 1))
                lv144: R.Tensor((1, 4, 64, 64), dtype="float32") = R.add(lv142, lv143)
                lv145: R.Tensor((1, 512, 64, 64), dtype="float32") = R.nn.conv2d(
                    lv144,
                    lv,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv146: R.Tensor((1, 512, 1, 1), dtype="float32") = R.reshape(lv1, (1, 512, 1, 1))
                lv147: R.Tensor((1, 512, 64, 64), dtype="float32") = R.add(lv145, lv146)
                gv: R.Tensor((1, 512, 64, 64), dtype="float32") = lv147
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            z: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w0: R.Tensor((512, 4, 3, 3), dtype="float16"),
            w1: R.Tensor((512,), dtype="float16"),
            w2: R.Tensor((4, 4, 1, 1), dtype="float16"),
            w3: R.Tensor((4,), dtype="float16"),
        ) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 64, 64), dtype="float16") = R.astype(z, dtype="float16")
                lv_1: R.Tensor((512, 4, 3, 3), dtype="float16") = w0
                lv1: R.Tensor((512,), dtype="float16") = w1
                lv140: R.Tensor((4, 4, 1, 1), dtype="float16") = w2
                lv141: R.Tensor((4,), dtype="float16") = w3
                lv1_1: R.Tensor((1, 4, 64, 64), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv140,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv142: R.Tensor((1, 4, 64, 64), dtype="float16") = R.astype(lv1_1, dtype="float16")
                lv143: R.Tensor((1, 4, 1, 1), dtype="float16") = R.reshape(lv141, (1, 4, 1, 1))
                lv144: R.Tensor((1, 4, 64, 64), dtype="float16") = R.add(lv142, lv143)
                lv2: R.Tensor((1, 512, 64, 64), dtype="float32") = R.nn.conv2d(
                    lv144,
                    lv_1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv145: R.Tensor((1, 512, 64, 64), dtype="float16") = R.astype(lv2, dtype="float16")
                lv146: R.Tensor((1, 512, 1, 1), dtype="float16") = R.reshape(lv1, (1, 512, 1, 1))
                lv147: R.Tensor((1, 512, 64, 64), dtype="float16") = R.add(lv145, lv146)
                gv: R.Tensor((1, 512, 64, 64), dtype="float32") = R.astype(lv147, dtype="float32")
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            z: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w0: R.Tensor((512, 4, 3, 3), dtype="float16"),
            w1: R.Tensor((512,), dtype="float16"),
            w2: R.Tensor((4, 4, 1, 1), dtype="float16"),
            w3: R.Tensor((4,), dtype="float16"),
        ) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 4, 64, 64), dtype="float16") = R.astype(z, dtype="float16")
                lv_1: R.Tensor((512, 4, 3, 3), dtype="float16") = w0
                lv1: R.Tensor((512,), dtype="float16") = w1
                lv140: R.Tensor((4, 4, 1, 1), dtype="float16") = w2
                lv141: R.Tensor((4,), dtype="float16") = w3
                lv142: R.Tensor((1, 4, 64, 64), dtype="float16") = R.nn.conv2d(
                    lv,
                    lv140,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                lv143: R.Tensor((1, 4, 1, 1), dtype="float16") = R.reshape(
                    lv141, R.shape([1, 4, 1, 1])
                )
                lv144: R.Tensor((1, 4, 64, 64), dtype="float16") = R.add(lv142, lv143)
                lv145: R.Tensor((1, 512, 64, 64), dtype="float16") = R.nn.conv2d(
                    lv144,
                    lv_1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
                lv146: R.Tensor((1, 512, 1, 1), dtype="float16") = R.reshape(
                    lv1, R.shape([1, 512, 1, 1])
                )
                lv147: R.Tensor((1, 512, 64, 64), dtype="float16") = R.add(lv145, lv146)
                gv: R.Tensor((1, 512, 64, 64), dtype="float32") = R.astype(lv147, dtype="float32")
                R.output(gv)
            return gv

    binding = {
        "w0": np.random.uniform(size=(512, 4, 3, 3)).astype("float16"),
        "w1": np.random.uniform(size=(512,)).astype("float16"),
        "w2": np.random.uniform(size=(4, 4, 1, 1)).astype("float16"),
        "w3": np.random.uniform(size=(4,)).astype("float16"),
    }
    binding = {k: tvm_tirx.runtime.tensor(v) for k, v in binding.items()}
    Input = relax.transform.BindParams("main", binding)(Input)
    Expected = relax.transform.BindParams("main", binding)(Expected)
    Expected2 = relax.transform.BindParams("main", binding)(Expected2)
    _assert_test(Input, Expected, Expected2)


def test_tuple_get():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(
            x: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w: R.Tensor((512, 4, 3, 3), dtype="float32"),
            bias: R.Tensor((512, 1, 1), dtype="float32"),
        ) -> R.Tensor((1, 256, 64, 64), dtype="float32"):
            with R.dataflow():
                conv = R.nn.conv2d(
                    x,
                    w,
                    strides=[1, 1],
                    padding=[0, 0, 1, 1],
                )
                bias_out = R.add(conv, bias)
                split = R.split(bias_out, indices_or_sections=2, axis=1)
                out = R.add(split[0], split[1])
                R.output(out)
            return out

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w: R.Tensor((512, 4, 3, 3), dtype="float32"),
            bias: R.Tensor((512, 1, 1), dtype="float32"),
        ):
            with R.dataflow():
                lv = R.astype(x, dtype="float16")
                lv1 = R.astype(w, dtype="float16")
                conv = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 1, 1],
                    out_dtype="float16",
                )
                lv2 = R.astype(conv, dtype="float32")
                bias_out = R.add(lv2, bias)
                split = R.split(bias_out, indices_or_sections=2, axis=1)
                lv3 = split[0]
                lv4 = split[1]
                out = R.add(lv3, lv4)
                R.output(out)
            return out

    _assert_test(Module, expected2=Expected)


def test_conv2d_bias_fp32():
    @tvm_tirx.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w: R.Tensor((512, 4, 3, 3), dtype="float32"),
            bias: R.Tensor((512,), dtype="float32"),
        ) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
            # block 0
            with R.dataflow():
                lv142: R.Tensor((1, 512, 62, 62), dtype="float32") = R.nn.conv2d(
                    x,
                    w,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    out_dtype="float32",
                )
                lv143: R.Tensor((1, 512, 1, 1), dtype="float32") = R.reshape(bias, (1, 512, 1, 1))
                lv144: R.Tensor((1, 512, 62, 62), dtype="float32") = R.add(lv142, lv143)
                R.output(lv144)
            return lv144

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w: R.Tensor((512, 4, 3, 3), dtype="float32"),
            bias: R.Tensor((512,), dtype="float32"),
        ) -> R.Tensor((1, 512, 62, 62), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 4, 64, 64), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((512, 4, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv142: R.Tensor((1, 512, 62, 62), dtype="float16") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    out_dtype="float16",
                )
                lv2: R.Tensor((512,), dtype="float16") = R.astype(bias, dtype="float16")
                lv143: R.Tensor((1, 512, 1, 1), dtype="float16") = R.reshape(
                    lv2, R.shape([1, 512, 1, 1])
                )
                lv3: R.Tensor((1, 512, 62, 62), dtype="float16") = R.add(lv142, lv143)
                lv144: R.Tensor((1, 512, 62, 62), dtype="float32") = R.astype(lv3, dtype="float32")
                R.output(lv144)
            return lv144

    @tvm_tirx.script.ir_module
    class Expected_no_bias_cast:
        @R.function
        def main(
            x: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w: R.Tensor((512, 4, 3, 3), dtype="float32"),
            bias: R.Tensor((512,), dtype="float32"),
        ) -> R.Tensor((1, 512, 62, 62), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 4, 64, 64), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((512, 4, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv142: R.Tensor((1, 512, 62, 62), dtype="float16") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    out_dtype="float16",
                )
                lv143: R.Tensor((1, 512, 1, 1), dtype="float32") = R.reshape(
                    bias, R.shape([1, 512, 1, 1])
                )
                lv2: R.Tensor((1, 512, 62, 62), dtype="float32") = R.astype(lv142, dtype="float32")
                lv144: R.Tensor((1, 512, 62, 62), dtype="float32") = R.add(lv2, lv143)
                R.output(lv144)
            return lv144

    binding_np = {
        "w": np.random.uniform(size=(512, 4, 3, 3)).astype("float32"),
        "bias": np.random.uniform(size=(512,)).astype("float32"),
    }
    binding = {k: tvm_tirx.runtime.tensor(v) for k, v in binding_np.items()}

    Input_bound = relax.transform.BindParams("main", binding)(Input)
    Expected = relax.transform.BindParams("main", binding)(Expected)

    _assert_test(Input_bound, expected2=Expected)

    binding_np["bias"][0] = 70000  # Out of fp16 range
    binding = {k: tvm_tirx.runtime.tensor(v) for k, v in binding_np.items()}
    Input_bound = relax.transform.BindParams("main", binding)(Input)
    Expected_no_bias_cast = relax.transform.BindParams("main", binding)(Expected_no_bias_cast)

    _assert_test(Input_bound, expected2=Expected_no_bias_cast)


def test_convert_sig():
    @tvm_tirx.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w: R.Tensor((512, 4, 3, 3), dtype="float32"),
            bias: R.Tensor((512,), dtype="float32"),
        ) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
            # block 0
            with R.dataflow():
                lv142: R.Tensor((1, 512, 62, 62), dtype="float32") = R.nn.conv2d(
                    x,
                    w,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    out_dtype="float32",
                )
                lv143: R.Tensor((1, 512, 1, 1), dtype="float32") = R.reshape(bias, (1, 512, 1, 1))
                lv144: R.Tensor((1, 512, 62, 62), dtype="float32") = R.add(lv142, lv143)
                R.output(lv144)
            return lv144

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 4, 64, 64), dtype="float32"),
            w: R.Tensor((512, 4, 3, 3), dtype="float16"),
            bias: R.Tensor((512,), dtype="float16"),
        ) -> R.Tensor((1, 512, 62, 62), dtype="float32"):
            with R.dataflow():
                lv = R.astype(x, dtype="float16")
                lv142 = R.nn.conv2d(
                    lv, w, strides=[1, 1], padding=[0, 0, 0, 0], out_dtype="float16"
                )
                lv143 = R.reshape(bias, R.shape([1, 512, 1, 1]))
                lv1 = R.add(lv142, lv143)
                lv144 = R.astype(lv1, dtype="float32")
                R.output(lv144)
            return lv144

    mod = ToMixedPrecision(out_dtype="float16", fp16_input_names=["w", "bias"])(Input)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_call_tir_with_float16_args():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([64], "float16")):
            cls = Before
            with R.dataflow():
                B = R.call_tir(cls.tir_identity, [A], ty_args=[R.Tensor([64], "float16")])
                C = R.call_tir(cls.tir_identity, [B], ty_args=[R.Tensor([64], "float16")])
                R.output(C)
            return C

        @Ts.function
        def tir_identity(
            Input: T.Tensor(64, "float16"),
            Output: T.Tensor(64, "float16"),
        ):
            for i in range(64):
                with Ts.sblock("copy"):
                    vi = Ts.axis.remap("S", [i])
                    Output[vi] = Input[vi]

    Expected = Before

    After = ToMixedPrecision()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_local_function_in_dataflow_block():
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((), "float32")):
            with R.dataflow():

                @R.function
                def double(y: R.Tensor((), "float32")) -> R.Tensor((), "float32"):
                    return R.add(y, y)

                gv = double(x)
                R.output(gv)
            return gv

    transformed = ToMixedPrecision()(Module)
    tvm_tirx.ir.assert_structural_equal(transformed, Module)


def test_dynamic_strided_slice():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            begin: R.Tensor((4,), "int64"),
            end: R.Tensor((4,), "int64"),
            strides: R.Tensor((4,), "int64"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv = R.dynamic_strided_slice(lv, begin, end, strides)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            begin: R.Tensor((4,), dtype="int64"),
            end: R.Tensor((4,), dtype="int64"),
            strides: R.Tensor((4,), dtype="int64"),
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
                lv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv3: R.Tensor((2, 4, 26, 26), dtype="float16") = R.astype(lv2, dtype="float16")
                lv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.astype(lv3, dtype="float32")
                gv: R.Tensor(None, dtype="float32", ndim=4) = R.dynamic_strided_slice(
                    lv4, begin, end, strides
                )
                R.output(gv)
            return gv

    _assert_test(Input, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
