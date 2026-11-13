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
# pylint: disable=invalid-name, unused-argument, missing-function-docstring, abstract-method
"""Relax Remove Redundant Reshape ops"""

import tvm_ffi

from tvm import relax
from tvm.relax import Expr
from tvm.relax.dpl import is_op, rewrite_call, wildcard
from tvm_tirx import IRModule
from tvm_tirx.ir.transform import PassContext

from . import function_pass


def _can_reparent(arg: Expr, output_shape: Expr) -> bool:
    """Whether ``reshape(arg, output_shape)`` still asks for ``output_shape``.

    A literal ``0`` in a reshape target means "copy the corresponding input dimension",
    and ``relax.op.reshape`` resolves it against the input it is handed. Moving such a
    target onto a different input therefore changes what it asks for. For
    ``x: (0, 3, 5)``, ``reshape(reshape(x, [0, 0, 5]), [0, 0, 0])`` is ``(0, 0, 0)``,
    while the combined ``reshape(x, [0, 0, 0])`` copies all three dimensions back and is
    ``(0, 3, 5)``.

    Rather than reason about which zeros are safe, re-resolve the target against the new
    input and keep the rewrite only when it comes back unchanged.
    """
    try:
        reparented = relax.op.reshape(arg, output_shape)
    except Exception:  # pylint: disable=broad-except
        # reshape cannot resolve a 0 or -1 without a known input shape.
        return False
    return tvm_ffi.structural_equal(reparented.args[1], output_shape)


@function_pass(opt_level=0)
class RemoveRedundantReshape:
    """
    Transformation pass to remove redundant reshape operator
    """

    def __init__(self):
        self.input1 = wildcard()
        shape1 = wildcard()
        pattern_redundant_reshape = is_op("relax.reshape")(self.input1, shape1)
        self.no_op_reshape = pattern_redundant_reshape
        shape2 = wildcard()
        self.repeated_reshape = is_op("relax.reshape")(pattern_redundant_reshape, shape2)
        self.pattern = self.repeated_reshape | self.no_op_reshape

    def transform_function(self, func: Expr, mod: IRModule, ctx: PassContext) -> IRModule:
        """
        Tarnsformation function to remove redundant reshape
        where tensors before and after reshape are of same dimentions.

        Parameters
        --------------
        func: Expr
            The relax function to be optimized

        mod: IRModule
            The IR module

        ctx: PassContext
            Relax pass context
        """

        updated_func = func

        # Skip primitive functions
        if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
            return updated_func

        def rewriter(expr, matches):
            arg = matches[self.input1]

            if self.repeated_reshape in matches:
                output_shape = matches[self.repeated_reshape].args[1]
                if _can_reparent(arg, output_shape):
                    return relax.op.reshape(arg, output_shape)
                return expr

            elif self.no_op_reshape in matches:
                output_shape = matches[self.no_op_reshape].args[1]
                if arg.ty.shape and tvm_ffi.structural_equal(arg.ty.shape, output_shape):
                    return arg
            return expr

        updated_func = rewrite_call(self.pattern, rewriter, func)

        return updated_func
