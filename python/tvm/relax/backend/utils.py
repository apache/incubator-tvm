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
# pylint: disable=invalid-name
"""Utils for BYOC pattern matching"""

from tvm import relax
from tvm.relax import DataflowVar, PyExprMutator
from tvm.relax.transform import PatternCheckContext
from tvm_tirx.target import Target


class BackendDispatcher(PyExprMutator):
    """Base class for backend dispatcher"""

    def __init__(self, mod, target: Target | None = None):
        super().__init__(mod)
        self.target = Target(target) if target is not None else None

    @staticmethod
    def is_gpu_target(target: Target) -> bool:
        """Check if the target is a GPU target."""
        return "gpu" in target.keys

    @staticmethod
    def get_shape_dtype(expr: relax.Expr) -> tuple[relax.ShapeExpr, str]:
        """Get shape and dtype from an expression.
        If the shape and dtype is unknown, raise an error."""
        ty = expr.ty
        if not isinstance(expr.ty, relax.TensorType):
            raise ValueError(f"Expecting a expr with TensorType, but got {expr} with {expr.ty}")

        shape, dtype = ty.shape, ty.dtype
        if shape is None:
            raise ValueError(
                f"Expecting a expr with known shape, but got {expr} with unknown shape"
            )

        return shape, dtype

    def _get_target(self, ty: relax.Type) -> Target:
        def find_target(ty):
            if isinstance(ty, relax.TensorType) and ty.vdevice is not None:
                return ty.vdevice.target
            if isinstance(ty, relax.TupleType):
                for field in ty.fields:
                    target = find_target(field)
                    if target is not None:
                        return target
            return None

        annotated_target = find_target(ty)
        if annotated_target is not None:
            return annotated_target
        # Fall back to the explicit compilation target
        target = self.target
        if target is None:
            raise ValueError(
                "Target not found. Please ensure that the target is annotated within the module, "
                "or pass an explicit target to the dispatch pass."
            )
        return target


def has_leaking_intermediate_variables(context: PatternCheckContext) -> bool:
    """
    Check whether intermediate variables in the region to be fused are used outside
    the fused region.
    """
    defined_vars = set(context.matched_bindings.keys())
    output_var = context.value_to_bound_var[context.matched_expr]
    intermediate_vars = {v for v in context.matched_bindings if v != output_var}

    if any(not isinstance(v, DataflowVar) for v in intermediate_vars):
        # If intermediate variable is not a DataflowVar, it can be accessed and potentially
        # used outside the DataflowBlock.
        return True

    # Check whether all users of an intermediate variable are inside the fused region.
    for var in intermediate_vars:
        if any(var_user not in defined_vars for var_user in context.var_usages[var]):
            return True

    return False
