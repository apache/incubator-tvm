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

import pytest
from utils import skip_unless_adreno_opencl_vulkan, verify_results

import tvm  # noqa: F401  # Register the parent Relax script dialect.
import tvm_tirx
import tvm_tirx.testing
from tvm.testing import device_enabled
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R

TARGETS = [
    tvm_tirx.target.Target("qcom/adreno-opencl-texture"),
    #    tvm.target.Target("qcom/adreno-vulkan-texture"),
]
ref_target = tvm_tirx.target.Target("llvm")


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 64, 56, 56), "float32"), w: R.Tensor((32, 64, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 32, 54, 54), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                R.output(gv)
            return gv

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_relu():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_relu_conv2d_relu():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                x0: R.Tensor((2, 16, 28, 28), "float32") = R.nn.relu(x)
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x0, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_relu_tanh():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 4, 26, 26), "float32") = R.tanh(gv2)
                R.output(gv3)
            return gv3

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_add():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_sum():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4), "float32") = R.sum(gv, axis=[2, 3])
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_sum_keepdims():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 1, 1), "float32") = R.sum(gv, axis=[2, 3], keepdims=True)
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_sum_reduce():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 26), "float32") = R.sum(gv, axis=[1, 2])
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_transpose():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv, axes=[3, 2, 1, 0])
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_expand_dims():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=6):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 1, 4, 1, 26, 26), "float32") = R.expand_dims(gv, axis=(-3, 1))
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_squeeze():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            with R.dataflow():
                gv: R.Tensor((1, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((4, 26, 26), "float32") = R.squeeze(gv, axis=[0])
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_strided_slice():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.strided_slice(
                    gv, begin=[0, 0, 0], end=[4, 26, 26], strides=[2, 3, 4], axes=[1, 2, 3]
                )
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_relu_concat():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                R.output(gv3)
            return gv3

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_relu_concat_split():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                gv4 = R.split(gv3, indices_or_sections=2, axis=1)
                # TODO @Siva: Multi value return have an issue at runtime.
                gv5 = gv4[0]
                R.output(gv5)
            return gv5

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_relu_concat_split_transpose_concat():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                gv4 = R.split(gv3, indices_or_sections=2, axis=1)
                gv5: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv4[0], axes=[3, 2, 1, 0])
                gv6: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv4[1], axes=[3, 2, 1, 0])
                gv7: R.Tensor((26, 26, 8, 2), "float32") = R.concat((gv5, gv6), axis=2)
                R.output(gv7)
            return gv7

    verify_results(Input, target, ref_target)


@pytest.mark.skip(reason="Known failure: numerical mismatch in texture lowering")
@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_maxpool2d():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.max_pool2d(
                    gv,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    padding=[0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.skip(reason="Known failure: numerical mismatch in texture lowering")
@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_avgpool2d():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.adaptive_avg_pool2d(gv, output_size=[13, 13], layout="NCHW")
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_softmax():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.softmax(gv, axis=1)
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_layernorm():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            gamma: R.Tensor((26, 26), dtype="float32"),
            beta: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.layer_norm(
                    gv, gamma, beta, axes=[-2, -1]
                )
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_binary_broadcast():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            bias: R.Tensor((26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_binary_ewise_scalar():
    target = TARGETS[0]

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, R.const(1, "float32"))
                R.output(gv2)
            return gv2

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_residual_block():
    target = TARGETS[0]
    r"""
    - some kind of residual block followed by convolution to have texture after residual block
    - scalar data type verification which should be mapped to global memory scope
        layout_transform (NCHW->NCHW4c)
                  |                      <- buffer
                conv2d (1)                  <- to get textures as output
               /         \
            conv2d (2)    |
                 \       /
                    add                     <- add should be fused into conv2d (2)
                multiply to scalar          <- buffer to the input of multiply scalar value
                    relu
                     |                      <- texture in intermediate tensor
                  conv2d (3)
                   relu
                     |                      <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((32, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 32, 1, 1), "float32"),
            w3: R.Tensor((32, 32, 2, 2), "float32"),
            bias: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[1, 1], out_dtype="float32")
                bias_1 = R.multiply(bias, R.const(0.15, "float32"))
                gv4 = R.add(gv3, bias_1)
                gv5 = R.nn.relu(gv4)
                gv6 = R.nn.conv2d(gv5, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.nn.relu(gv6)
                R.output(gv7)
            return gv7

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_conv2d_fallback_to_buffer_conv2d():
    target = TARGETS[0]
    r"""
        layout_transform (NCHW->NCHW4c)
                  |                      <- texture
                conv2d (1)               <- textures as output
               /         \
            conv2d (2)    conv2d (3)     <- conv2d (2) emits texture, conv2d (3) emits buffer
                 \       /               <- concat shouldn't support textures here
                concatenation
                     |                   <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((96, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 96, 2, 2), "float32"),
            w3: R.Tensor((5, 96, 2, 2), "float32"),
            bias1: R.Tensor((1, 96, 1, 1), "float32"),
            bias2: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias1)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[2, 2], out_dtype="float32")
                gv6 = R.nn.conv2d(gv2, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.concat((gv3, gv6), axis=1)
                R.output(gv7)
            return gv7

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_conv2d_conv2d_conv2d_concat():
    target = TARGETS[0]
    r"""
        layout_transform (NCHW->NCHW4c)
                  |                      <- texture
                conv2d (1)               <- textures as output
               /         \
            conv2d (2)    conv2d (3)
                 \       /               <- concat does support textures here
                concatenation
                     |                   <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((96, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 96, 2, 2), "float32"),
            w3: R.Tensor((8, 96, 2, 2), "float32"),
            bias1: R.Tensor((1, 96, 1, 1), "float32"),
            bias2: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias1)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[2, 2], out_dtype="float32")
                gv6 = R.nn.conv2d(gv2, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.concat((gv3, gv6), axis=1)
                R.output(gv7)
            return gv7

    verify_results(Input, target, ref_target)


@pytest.mark.skip(reason="Known failure: numerical mismatch in texture lowering")
@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_pooling_branching_texture_params():
    target = TARGETS[0]
    r"""
    Verification of the pooling and many branches having textures
                layout_transform (NCHW->NCHW4c)
                         |                        <- texture
                      conv2d (0)                  <- to get textures
                         |                        <- textures
                     pooling
               /           \           \          <- textures
            conv2d (1)    conv2d (2)    conv2d (3)
                \             /           |
                     add                  |       <- to have  the only one output, will be fused
                      \                  /
                            add                  <- to have  the only one output, will be fused
                             |                   <- buffer
                    layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((32, 32, 1, 1), "float32"),
            w2: R.Tensor((32, 32, 2, 2), "float32"),
            w3: R.Tensor((32, 32, 1, 1), "float32"),
            w4: R.Tensor((32, 32, 2, 2), "float32"),
            bias1: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[1, 1], out_dtype="float32")
                gv1 = R.nn.max_pool2d(gv, pool_size=[2, 2], strides=[2, 2])
                gv2 = R.nn.conv2d(
                    gv1, w2, padding=[0, 0, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                gv5 = R.nn.conv2d(
                    gv1, w3, padding=[0, 0, 0, 0], strides=[1, 1], out_dtype="float32"
                )
                gv6 = R.nn.conv2d(
                    gv1, w4, padding=[0, 1, 1, 0], strides=[1, 1], out_dtype="float32"
                )
                gv8 = R.add(gv2, gv5)
                gv9 = R.add(gv8, gv6)
                R.output(gv9)
            return gv9

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_injective_inputs1():
    target = TARGETS[0]
    r"""
                                     Input
                               /                   \
                            /                      |
                         |                        /
                      conv2d (1)                 /
                         |                      /
                      conv2d (2)              mean
                  /         \                 /
                 |           |      \        /
                 |           |       (3) add
                 |           |         |
                 |             \    /
                 \                mul
                  \            /
                        add

    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 4, 40, 40), "float32"),
            w1: R.Tensor((4, 4, 3, 3), "float32"),
            w2: R.Tensor((4, 4, 3, 3), "float32"),
            w3: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                mean = R.mean(x, axis=1, keepdims=True)
                conv1 = R.nn.conv2d(
                    x, w1, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                ad3 = R.add(conv1, conv2)
                ad1 = R.add(mean, conv1)
                ad2 = R.multiply(ad1, conv2)
                gv = R.add(ad3, ad2)
                R.output(gv)
            return gv

    verify_results(Input, target, ref_target)


@pytest.mark.gpu
@skip_unless_adreno_opencl_vulkan
@pytest.mark.skipif(not device_enabled(TARGETS[0]), reason="opencl not enabled")
def test_injective_nwo_inputs2():
    target = TARGETS[0]
    r"""
                                     Input
                               /             \
                         |                    \
                      conv2d                   \
                         |                     /
                      conv2d           mean    /
                  /         \                 /
                add         |   \             |
                 |           |    \           |
                 |           |      \        /
                 |           |       (3) add
                 |           |         |
                 |            \       /
                 |             \    /
                 \                mul
                  \            /
                        add

    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 4, 40, 40), "float32"),
            w1: R.Tensor((4, 4, 3, 3), "float32"),
            w2: R.Tensor((4, 4, 3, 3), "float32"),
            w3: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                mean = R.mean(x, axis=1, keepdims=True)
                conv1 = R.nn.conv2d(
                    x, w1, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                ad3 = R.add(conv1, conv2)
                ad1 = R.add(mean, conv1)
                ad2 = R.multiply(ad1, conv2)
                gv = R.add(ad2, ad3)
                R.output(gv)
            return gv

    verify_results(Input, target, ref_target)


if __name__ == "__main__":
    tvm_tirx.testing.main()
