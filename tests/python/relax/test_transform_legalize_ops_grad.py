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
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_nll_loss_backward():
    # fmt: off
    @tvm_tirx.script.ir_module
    class NLLLossBackward:
        @R.function
        def main(output_grad: R.Tensor((), "float32"), predictions: R.Tensor((2, 3, 4, 5), "float32"), targets: R.Tensor((2, 4, 5), "int64"), weights: R.Tensor((4,), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.grad.nll_loss_backward(output_grad, predictions, targets, weights, reduction="mean", ignore_index=-1)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def nll_loss_backward(rxplaceholder: T.Tensor((), "float32"), rxplaceholder_1: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_2: T.Tensor((T.int64(2), T.int64(4), T.int64(5)), "int64"), rxplaceholder_3: T.Tensor((T.int64(4),), "float32"), pred_grad: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            all_weights = Ts.sblock_alloc_tensor((T.int64(2), T.int64(4), T.int64(5)))
            T_broadcast_to = Ts.sblock_alloc_tensor((T.int64(2), T.int64(4), T.int64(5)))
            all_weights_red = Ts.sblock_alloc_tensor(())
            T_divide = Ts.sblock_alloc_tensor((T.int64(2), T.int64(4), T.int64(5)))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("all_weights"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder_3[rxplaceholder_2[v_i0, v_i1, v_i2]], rxplaceholder_2[v_i0, v_i1, v_i2])
                    Ts.writes(all_weights[v_i0, v_i1, v_i2])
                    all_weights[v_i0, v_i1, v_i2] = rxplaceholder_3[rxplaceholder_2[v_i0, v_i1, v_i2]]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("T_broadcast_to"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                    T_broadcast_to[v_ax0, v_ax1, v_ax2] = rxplaceholder[()]
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("all_weights_red"):
                    v_k0, v_k1, v_k2 = Ts.axis.remap("RRR", [k0, k1, k2])
                    Ts.reads(all_weights[v_k0, v_k1, v_k2])
                    Ts.writes(all_weights_red[()])
                    with Ts.init():
                        all_weights_red[()] = T.float32(0)
                    all_weights_red[()] = all_weights_red[()] + all_weights[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("T_divide"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_broadcast_to[v_ax0, v_ax1, v_ax2], all_weights_red[()])
                    Ts.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = T_broadcast_to[v_ax0, v_ax1, v_ax2] / all_weights_red[()]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("pred_grad"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_2[v_i0, v_i2, v_i3], all_weights[v_i0, v_i2, v_i3], T_divide[v_i0, v_i2, v_i3])
                    Ts.writes(pred_grad[v_i0, v_i1, v_i2, v_i3])
                    pred_grad[v_i0, v_i1, v_i2, v_i3] = T.Select(v_i1 == rxplaceholder_2[v_i0, v_i2, v_i3], all_weights[v_i0, v_i2, v_i3] * T.float32(-1) * T_divide[v_i0, v_i2, v_i3], T.float32(0))

        @R.function
        def main(output_grad: R.Tensor((), dtype="float32"), predictions: R.Tensor((2, 3, 4, 5), dtype="float32"), targets: R.Tensor((2, 4, 5), dtype="int64"), weights: R.Tensor((4,), dtype="float32")) -> R.Tensor((2, 3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.nll_loss_backward, (output_grad, predictions, targets, weights), ty_args=[R.Tensor((2, 3, 4, 5), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(NLLLossBackward)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_nll_loss_backward_no_weight():
    # fmt: off
    @I.ir_module
    class NLLLossBackward:
        @R.function
        def main(output_grad: R.Tensor((), "float32"), predictions: R.Tensor((2, 3, 4, 5), "float32"), targets: R.Tensor((2, 4, 5), "int64")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.grad.nll_loss_backward(output_grad, predictions, targets, reduction="mean", ignore_index=-1)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def te_nll_loss_backward_no_weight(rxplaceholder: T.Tensor((), "float32"), rxplaceholder_1: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_2: T.Tensor((T.int64(2), T.int64(4), T.int64(5)), "int64"), pred_grad: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_full = Ts.sblock_alloc_tensor((T.int64(3),))
            all_weights = Ts.sblock_alloc_tensor((T.int64(2), T.int64(4), T.int64(5)))
            T_broadcast_to = Ts.sblock_alloc_tensor((T.int64(2), T.int64(4), T.int64(5)))
            all_weights_red = Ts.sblock_alloc_tensor(())
            T_divide = Ts.sblock_alloc_tensor((T.int64(2), T.int64(4), T.int64(5)))
            for ax0 in range(T.int64(3)):
                with Ts.sblock("T_full"):
                    v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                    Ts.reads()
                    Ts.writes(T_full[v_ax0])
                    T_full[v_ax0] = T.float32(1)
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("all_weights"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(T_full[rxplaceholder_2[v_i0, v_i1, v_i2]], rxplaceholder_2[v_i0, v_i1, v_i2])
                    Ts.writes(all_weights[v_i0, v_i1, v_i2])
                    all_weights[v_i0, v_i1, v_i2] = T_full[rxplaceholder_2[v_i0, v_i1, v_i2]]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("T_broadcast_to"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                    T_broadcast_to[v_ax0, v_ax1, v_ax2] = rxplaceholder[()]
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("all_weights_red"):
                    v_k0, v_k1, v_k2 = Ts.axis.remap("RRR", [k0, k1, k2])
                    Ts.reads(all_weights[v_k0, v_k1, v_k2])
                    Ts.writes(all_weights_red[()])
                    with Ts.init():
                        all_weights_red[()] = T.float32(0)
                    all_weights_red[()] = all_weights_red[()] + all_weights[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("T_divide"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_broadcast_to[v_ax0, v_ax1, v_ax2], all_weights_red[()])
                    Ts.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = T_broadcast_to[v_ax0, v_ax1, v_ax2] / all_weights_red[()]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("pred_grad"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_2[v_i0, v_i2, v_i3], all_weights[v_i0, v_i2, v_i3], T_divide[v_i0, v_i2, v_i3])
                    Ts.writes(pred_grad[v_i0, v_i1, v_i2, v_i3])
                    pred_grad[v_i0, v_i1, v_i2, v_i3] = T.Select(v_i1 == rxplaceholder_2[v_i0, v_i2, v_i3], all_weights[v_i0, v_i2, v_i3] * T.float32(-1) * T_divide[v_i0, v_i2, v_i3], T.float32(0))

        @R.function
        def main(output_grad: R.Tensor((), dtype="float32"), predictions: R.Tensor((2, 3, 4, 5), dtype="float32"), targets: R.Tensor((2, 4, 5), dtype="int64")) -> R.Tensor((2, 3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.te_nll_loss_backward_no_weight, (output_grad, predictions, targets), ty_args=[R.Tensor((2, 3, 4, 5), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(NLLLossBackward)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_nll_loss_backward_no_batch():
    # fmt: off
    @tvm_tirx.script.ir_module
    class NLLLossBackward:
        @R.function
        def main(output_grad: R.Tensor((), "float32"), predictions: R.Tensor((4,), "float32"), targets: R.Tensor((), "int64"), weights: R.Tensor((4,), "float32")) -> R.Tensor((4,), "float32"):
            gv: R.Tensor((4,), "float32") = R.grad.nll_loss_backward(output_grad, predictions, targets, weights, reduction="mean", ignore_index=-1)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(output_grad: R.Tensor((), dtype="float32"), predictions: R.Tensor((4,), dtype="float32"), targets: R.Tensor((), dtype="int64"), weights: R.Tensor((4,), dtype="float32")) -> R.Tensor((4,), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.nll_loss_backward, (output_grad, predictions, targets, weights), ty_args=[R.Tensor((4,), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def nll_loss_backward(rxplaceholder: T.Tensor((), "float32"), rxplaceholder_1: T.Tensor((T.int64(4),), "float32"), rxplaceholder_2: T.Tensor((), "int64"), rxplaceholder_3: T.Tensor((T.int64(4),), "float32"), pred_grad: T.Tensor((T.int64(4),), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            all_weights = Ts.sblock_alloc_tensor(())
            T_broadcast_to = Ts.sblock_alloc_tensor(())
            T_divide = Ts.sblock_alloc_tensor(())
            with Ts.sblock("all_weights"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(rxplaceholder_3[rxplaceholder_2[()]], rxplaceholder_2[()])
                Ts.writes(all_weights[()])
                all_weights[()] = rxplaceholder_3[rxplaceholder_2[()]]
            with Ts.sblock("T_broadcast_to"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(rxplaceholder[()])
                Ts.writes(T_broadcast_to[()])
                T_broadcast_to[()] = rxplaceholder[()]
            with Ts.sblock("T_divide"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_broadcast_to[()], all_weights[()])
                Ts.writes(T_divide[()])
                T_divide[()] = T_broadcast_to[()] / all_weights[()]
            for i_index in range(T.int64(4)):
                with Ts.sblock("pred_grad"):
                    v_i = Ts.axis.spatial(T.int64(4), i_index)
                    Ts.reads(rxplaceholder_2[()], all_weights[()], T_divide[()])
                    Ts.writes(pred_grad[v_i])
                    pred_grad[v_i] = T.Select(v_i == rxplaceholder_2[()], all_weights[()] * T.float32(-1) * T_divide[()], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(NLLLossBackward)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_max_pool2d_backward():
    # fmt: off
    @tvm_tirx.script.ir_module
    class MaxPool2DBackward:
        @R.function
        def main(output_grad: R.Tensor((3, 2, 6, 5), "float32"), data: R.Tensor((3, 2, 10, 10), "float32")):
            gv = R.grad.max_pool2d_backward(output_grad, data, (5, 5), (2, 2), (2, 1, 2, 1), (1, 1), True)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def max_pool2d_backward(A: T.Tensor((T.int64(3), T.int64(2), T.int64(6), T.int64(5)), "float32"), B: T.Tensor((T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32"), T_pool_grad: T.Tensor((T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            pad_temp = Ts.sblock_alloc_tensor((T.int64(3), T.int64(2), T.int64(15), T.int64(13)))
            maxpool_grad_argmax_v0 = Ts.sblock_alloc_tensor((T.int64(3), T.int64(2), T.int64(6), T.int64(5)), "int64")
            maxpool_grad_argmax_v1 = Ts.sblock_alloc_tensor((T.int64(3), T.int64(2), T.int64(6), T.int64(5)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(3), T.int64(2), T.int64(15), T.int64(13)):
                with Ts.sblock("pad_temp"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(B[v_ax0, v_ax1, v_ax2 - T.int64(2), v_ax3 - T.int64(1)])
                    Ts.writes(pad_temp[v_ax0, v_ax1, v_ax2, v_ax3])
                    pad_temp[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(2) <= v_ax2 and v_ax2 < T.int64(12) and T.int64(1) <= v_ax3 and v_ax3 < T.int64(11), B[v_ax0, v_ax1, v_ax2 - T.int64(2), v_ax3 - T.int64(1)], T.float32(-3.4028234663852886e+38))
            for ax0, ax1, ax2, ax3, dh, dw in T.grid(T.int64(3), T.int64(2), T.int64(6), T.int64(5), T.int64(5), T.int64(5)):
                with Ts.sblock("maxpool_grad_argmax"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_dh, v_dw = Ts.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, dh, dw])
                    Ts.reads(pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw])
                    Ts.writes(maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3], maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3] = T.int64(-1)
                        maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(-3.4028234663852886e+38)
                    v_maxpool_grad_argmax_v0: T.let[T.int64] = T.Select(maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] > pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw] or (maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] == pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw] and maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3] < v_ax0 * T.int64(390) + v_ax1 * T.int64(195) + v_ax2 * T.int64(26) + v_dh * T.int64(13) + v_ax3 * T.int64(2) + v_dw), maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3], v_ax0 * T.int64(390) + v_ax1 * T.int64(195) + v_ax2 * T.int64(26) + T.Cast("int64", v_dh) * T.int64(13) + v_ax3 * T.int64(2) + T.Cast("int64", v_dw))
                    v_maxpool_grad_argmax_v1: T.let[T.float32] = T.Select(maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] > pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw], maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3], pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw])
                    maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3] = v_maxpool_grad_argmax_v0
                    maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] = v_maxpool_grad_argmax_v1
            for ax0, ax1, ax2, ax3, wh, ww in T.grid(T.int64(3), T.int64(2), T.int64(10), T.int64(10), T.int64(3), T.int64(3)):
                with Ts.sblock("T_pool_grad"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_wh, v_ww = Ts.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, wh, ww])
                    Ts.reads(maxpool_grad_argmax_v0[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww], A[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww])
                    Ts.writes(T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] + T.if_then_else(T.Select(v_ax2 < T.int64(3), T.int64(0), T.Div(v_ax2 - T.int64(3), T.int64(2)) + T.int64(1)) <= T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh and T.Select(v_ax3 < T.int64(4), T.int64(0), T.Div(v_ax3 - T.int64(4), T.int64(2)) + T.int64(1)) <= T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww and maxpool_grad_argmax_v0[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww] == v_ax0 * T.int64(390) + v_ax1 * T.int64(195) + v_ax2 * T.int64(13) + v_ax3 + T.int64(27), A[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww], T.float32(0))

        @R.function
        def main(output_grad: R.Tensor((3, 2, 6, 5), dtype="float32"), data: R.Tensor((3, 2, 10, 10), dtype="float32")) -> R.Tensor((3, 2, 10, 10), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.max_pool2d_backward, (output_grad, data), ty_args=[R.Tensor((3, 2, 10, 10), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(MaxPool2DBackward)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_avg_pool2d_backward():
    # fmt: off
    @tvm_tirx.script.ir_module
    class AvgPool2DBackward:
        @R.function
        def main(output_grad: R.Tensor((3, 2, 6, 5), "float32"), data: R.Tensor((3, 2, 10, 10), "float32")):
            gv = R.grad.avg_pool2d_backward(output_grad, data, (5, 5), (2, 2), (2, 1, 2, 1), (1, 1), True)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def avg_pool2d_backward(output_grad: T.Tensor((T.int64(3), T.int64(2), T.int64(6), T.int64(5)), "float32"), data: T.Tensor((T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32"), T_pool_grad: T.Tensor((T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3, wh, ww in T.grid(T.int64(3), T.int64(2), T.int64(10), T.int64(10), T.int64(3), T.int64(3)):
                with Ts.sblock("T_pool_grad"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_wh, v_ww = Ts.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, wh, ww])
                    Ts.reads(output_grad[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww])
                    Ts.writes(T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] + T.if_then_else(T.Select(v_ax2 < T.int64(3), T.int64(0), T.Div(v_ax2 - T.int64(3), T.int64(2)) + T.int64(1)) <= T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh and T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh < T.int64(6) and T.Select(v_ax3 < T.int64(4), T.int64(0), T.Div(v_ax3 - T.int64(4), T.int64(2)) + T.int64(1)) <= T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww and T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww < T.int64(5), output_grad[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww] / T.Cast("float32", T.max((T.min(T.Div(v_ax2 + T.int64(2), T.int64(2)) * T.int64(2) + T.int64(3) - T.Cast("int64", v_wh) * T.int64(2), T.int64(10)) - T.max(T.Div(v_ax2 + T.int64(2), T.int64(2)) - T.Cast("int64", v_wh) - T.int64(1), T.int64(0)) * T.int64(2)) * (T.min(T.Div(v_ax3 + T.int64(1), T.int64(2)) * T.int64(2) + T.int64(4) - T.Cast("int64", v_ww) * T.int64(2), T.int64(10)) - T.max(T.Div(v_ax3 + T.int64(1), T.int64(2)) * T.int64(2) - T.Cast("int64", v_ww) * T.int64(2) - T.int64(1), T.int64(0))), T.int64(1))), T.float32(0.0))

        @R.function
        def main(output_grad: R.Tensor((3, 2, 6, 5), dtype="float32"), data: R.Tensor((3, 2, 10, 10), dtype="float32")) -> R.Tensor((3, 2, 10, 10), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.avg_pool2d_backward, (output_grad, data), ty_args=[R.Tensor((3, 2, 10, 10), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(AvgPool2DBackward)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_take_backward():
    # fmt: off
    @tvm_tirx.script.ir_module
    class TakeBackward:
        @R.function
        def main(output_grad: R.Tensor((3, 2, 5), "float32"), x: R.Tensor((3, 4, 5), "float32"), indices: R.Tensor((2,), "int32")):
            gv = R.grad.take_backward(output_grad, x, indices, axis=1)
            return gv

    expected_rxplaceholder_offset = T.dynamic("expected_rxplaceholder_offset", "int64")
    expected_rxplaceholder_1_offset = T.dynamic("expected_rxplaceholder_1_offset", "int64")
    expected_rxplaceholder_2_offset = T.dynamic("expected_rxplaceholder_2_offset", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def take_backward(rxplaceholder: T.Tensor((T.int64(3), T.int64(2), T.int64(5)), "float32",  layout=T.TileLayout(T.S[(T.int64(3), T.int64(2), T.int64(5))] + (expected_rxplaceholder_offset))), rxplaceholder_1: T.Tensor((T.int64(3), T.int64(4), T.int64(5)), "float32",  layout=T.TileLayout(T.S[(T.int64(3), T.int64(4), T.int64(5))] + (expected_rxplaceholder_1_offset))), rxplaceholder_2: T.Tensor((T.int64(2),), 'int32',  layout=T.TileLayout(T.S[(T.int64(2),)] + (expected_rxplaceholder_2_offset))), out_buf: T.Tensor((T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})

            with Ts.sblock("take_backward"):
                for i_index in range(T.int64(60)):
                    out_buf[i_index // T.int64(5) // T.int64(4), i_index // T.int64(5) % T.int64(4), i_index % T.int64(5)] = T.float32(0)
                for parallel, serial in T.grid(T.int64(15), T.int64(2)):
                    out_buf[(parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) // T.int64(5) // T.int64(4), (parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) // T.int64(5) % T.int64(4), (parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) % T.int64(5)] = out_buf[(parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) // T.int64(5) // T.int64(4), (parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) // T.int64(5) % T.int64(4), (parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) % T.int64(5)] + rxplaceholder[(parallel // T.int64(5) * T.int64(5) * T.int64(2) + serial * T.int64(5) + parallel % T.int64(5)) // T.int64(5) // T.int64(2), (parallel // T.int64(5) * T.int64(5) * T.int64(2) + serial * T.int64(5) + parallel % T.int64(5)) // T.int64(5) % T.int64(2), (parallel // T.int64(5) * T.int64(5) * T.int64(2) + serial * T.int64(5) + parallel % T.int64(5)) % T.int64(5)]

        @R.function
        def main(output_grad: R.Tensor((3, 2, 5), dtype="float32"), x: R.Tensor((3, 4, 5), dtype="float32"), indices: R.Tensor((2,), dtype="int32")) -> R.Tensor((3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.take_backward, (output_grad, x, indices), ty_args=[R.Tensor((3, 4, 5), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(TakeBackward)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_take_backward_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    i = T.dynamic("i", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class TakeBackward:
        @R.function
        def main(output_grad: R.Tensor((m, i), "float32"), x: R.Tensor((m, n), "float32"), indices: R.Tensor((i,), "int32")):
            gv = R.grad.take_backward(output_grad, x, indices, axis=1)
            return gv

    m_take_backward = T.dynamic("m", "int64")
    i_take_backward = T.dynamic("i", "int64")
    n_take_backward = T.dynamic("n", "int64")
    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    i_main = T.dynamic("i", "int64")

    expected_rxplaceholder_offset_1 = T.dynamic("expected_rxplaceholder_offset_1", "int64")
    expected_rxplaceholder_1_offset_1 = T.dynamic("expected_rxplaceholder_1_offset_1", "int64")
    expected_rxplaceholder_2_offset_1 = T.dynamic("expected_rxplaceholder_2_offset_1", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def take_backward(rxplaceholder: T.Tensor((m_take_backward, i_take_backward), "float32",  layout=T.TileLayout(T.S[(m_take_backward, i_take_backward)] + (expected_rxplaceholder_offset_1))), rxplaceholder_1: T.Tensor((m_take_backward, n_take_backward), "float32",  layout=T.TileLayout(T.S[(m_take_backward, n_take_backward)] + (expected_rxplaceholder_1_offset_1))), rxplaceholder_2: T.Tensor((i_take_backward,), 'int32',  layout=T.TileLayout(T.S[(i_take_backward,)] + (expected_rxplaceholder_2_offset_1))), out_buf: T.Tensor((m_take_backward, n_take_backward), "float32")):
            T.func_attr({"tirx.noalias": True})

            with Ts.sblock("take_backward"):
                for i_1 in range(m_take_backward * n_take_backward):
                    out_buf[i_1 // n_take_backward, i_1 % n_take_backward] = T.float32(0)
                for parallel, serial in T.grid(m_take_backward, i_take_backward):
                    out_buf[(parallel * n_take_backward + T.Cast("int64", rxplaceholder_2[serial])) // n_take_backward, (parallel * n_take_backward + T.Cast("int64", rxplaceholder_2[serial])) % n_take_backward] = out_buf[(parallel * n_take_backward + T.Cast("int64", rxplaceholder_2[serial])) // n_take_backward, (parallel * n_take_backward + T.Cast("int64", rxplaceholder_2[serial])) % n_take_backward] + rxplaceholder[(parallel * i_take_backward + serial) // i_take_backward, (parallel * i_take_backward + serial) % i_take_backward]

        @R.function
        def main(output_grad: R.Tensor((m_main, i_main), dtype="float32"), x: R.Tensor((m_main, n_main), dtype="float32"), indices: R.Tensor((i_main,), dtype="int32")) -> R.Tensor((m_main, n_main), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.take_backward, (output_grad, x, indices), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(TakeBackward)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
