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
"""Relax Optimize Layout Transform pass."""

import tvm_ffi

from tvm.relax import Expr
from tvm.relax.dpl import TuplePattern, is_op, rewrite_call, wildcard
from tvm_tirx.ir.module import IRModule
from tvm_tirx.ir.transform import PassContext

from . import function_pass


@function_pass(opt_level=0)
class OptimizeLayoutTransform:
    """
    Pass to remove redundant transform layout operators
    introduced by AlterOpImpl pass.
    """

    def __init__(self):
        self.input = wildcard()
        pattern_transform_layout = is_op("relax.layout_transform")(self.input)
        pattern_1 = is_op("relax.layout_transform")(pattern_transform_layout)

        self.gv_ = wildcard()
        args = TuplePattern([pattern_transform_layout])
        pattern_2 = is_op("relax.call_tir")(self.gv_, args)
        self.pattern_2 = is_op("relax.layout_transform")(pattern_2)

        self.pattern = pattern_1 | self.pattern_2

    def transform_function(self, func: Expr, mod: IRModule, ctx: PassContext) -> IRModule:
        """
        Tranformation function to pattern match layout_transform -> layout_transform
        pattern

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
        updated_func = func

        # Skip primitive functions
        if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
            return updated_func

        def rewriter(expr, matches):
            arg1 = matches[self.pattern]
            if self.pattern_2 not in matches.keys():
                arg2 = matches[self.input]
            else:
                arg2 = matches[self.gv_]
                if "remove_pad" == self.mod[arg2].attrs["operator_name"]:
                    arg2 = matches[self.input]
            if hasattr(arg1.ty, "shape") and hasattr(arg2.ty, "shape"):
                if tvm_ffi.structural_equal(arg1.ty.shape, arg2.ty.shape):
                    return arg2
            return expr

        updated_func = rewrite_call(self.pattern, rewriter, func)

        return updated_func
