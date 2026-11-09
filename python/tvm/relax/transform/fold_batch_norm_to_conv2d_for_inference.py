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
# pylint: disable=invalid-name, unused-argument, redefined-argument-from-local
"""Relax Fold Batchnorm into Conv2D."""

import tvm_tirx
from tvm import relax
from tvm.relax import Expr
from tvm.relax.dpl import TupleGetItemPattern, is_const, is_op, rewrite_call, wildcard
from tvm_tirx.ir.module import IRModule
from tvm_tirx.ir.transform import PassContext

from . import function_pass


@function_pass(opt_level=0)
class FoldBatchnormToConv2D:
    """
    Fuse Batchnorm to its previous Conv2D
    This optimization is a special case of FoldScaleAxis that folds scale into conv2d weights.
    This pass can be removed when FoldScaleAcis enhances to support this case.
    """

    def __init__(self):
        self.input = wildcard()
        self.weight = is_const()
        self.pattern_conv2d = is_op("relax.nn.conv2d")(self.input, self.weight)
        self.bn_weight = is_const()
        self.bias = is_const()
        self.mean = is_const()
        self.variance = is_const()
        self.pattern_bn = is_op("relax.nn.batch_norm")(
            self.pattern_conv2d, self.bn_weight, self.bias, self.mean, self.variance
        )

        self.pattern = TupleGetItemPattern(self.pattern_bn, 0)

    def transform_function(self, func: Expr, mod: IRModule, ctx: PassContext) -> IRModule:
        """
        Tranformation function for pattern Conv2D+BatchNorm+TupleGetItem pattern
        Parameters
        ----------
        func: Expr
            The relax function to be optimized
        mod: IRModule
            The ir module
        ctx: PassContext
            Relax pass context
        """

        self.mod = mod
        updated_call = func

        # Skip primitive functions
        if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
            return updated_call

        def rewriter(expr, matches):
            conv_input = matches[self.input]
            conv_weight = matches[self.weight]
            bn_weight = matches[self.bn_weight]
            bn_bias = matches[self.bias]
            bn_mean = matches[self.mean]
            bn_variance = matches[self.variance]
            conv_op = matches[self.pattern_conv2d]
            bn_op = matches[self.pattern_bn]
            conv_attrs = conv_op.attrs
            bn_attrs = bn_op.attrs

            if not bn_attrs["scale"] or not bn_attrs["center"]:
                return expr
            out_layout = conv_attrs["out_layout"] or conv_attrs["data_layout"]
            if out_layout != "NCHW" or bn_attrs["axis"] % 4 != 1:
                return expr
            if conv_weight.ty.dtype != conv_op.ty.dtype:
                return expr

            bn_variance = relax.op.add(
                bn_variance,
                relax.prim_value(tvm_tirx.ir.FloatImm("float32", bn_attrs["epsilon"])),
            )
            dino = relax.op.sqrt(bn_variance)
            wt = relax.op.divide(bn_weight, dino)
            bs = relax.op.subtract(bn_bias, relax.op.multiply(bn_mean, wt))
            if conv_attrs["kernel_layout"] == "OIHW":
                wt = relax.op.reshape(wt, shape=(bn_weight.ty.shape[0], 1, 1, 1))
            elif conv_attrs["kernel_layout"] == "IOHW":
                wt = wt.reshape(1, bn_weight.ty.shape[0], 1, 1)
            else:
                return expr
            wt_conv = relax.op.multiply(conv_weight, wt)
            bs_args = relax.op.reshape(bs, shape=(1, bn_bias.ty.shape[0], 1, 1))

            conv_out = relax.Call(conv_op.op, (conv_input, wt_conv), conv_attrs)
            return relax.op.add(conv_out, bs_args)

        updated_call = rewrite_call(self.pattern, rewriter, func)

        return updated_call
