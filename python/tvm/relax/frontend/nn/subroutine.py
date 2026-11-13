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
# pylint: disable=too-many-lines,invalid-name,protected-access
"""nn.Module mixin for subroutine dispatch"""

import collections
import contextlib
import functools
import inspect
import re
import typing

import tvm_ffi

from tvm import relax
from tvm.relax.frontend import nn
from tvm_tirx import ir


def _camel_to_snake(name):
    """Convert from CamelCase to snake_case"""

    # Adapted from https://stackoverflow.com/a/1176023
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.lower()
    return name


def _normalize_expr(block_builder, arg, as_relax_expr=False):
    """Ensure that an argument is a relax.Expr with type"""
    if isinstance(arg, tuple):
        arg = relax.Tuple([_normalize_expr(block_builder, element) for element in arg])

    if isinstance(arg, relax.Expr) and isinstance(arg.ty, ir.MissingType):
        arg = block_builder.emit(arg)

    if isinstance(arg, nn.Tensor) and as_relax_expr:
        arg = arg._expr

    return arg


def _get_ty(arg):
    if isinstance(arg, relax.Expr):
        return arg.ty
    elif isinstance(arg, nn.Tensor):
        return arg._expr.ty
    elif isinstance(arg, tuple | list | tvm_ffi.Array):
        return relax.TupleType([_get_ty(field) for field in arg])
    else:
        raise TypeError(f"Cannot find type for {arg} of type {type(arg)}")


class SubroutineMixin:
    """A mixin that generates a

    Contains common logic for `tvm.relax.frontend.nn.Module` and
    `tvm.relax.testing.nn.Module`.
    """

    define_subroutine: bool = False

    def __init_subclass__(cls):
        """Update the cls.forward of subclasses"""
        if hasattr(cls, "forward"):
            is_wrapped = getattr(cls.forward, "_is_subroutine_mixin", False)
            if not is_wrapped:
                cls.forward = cls._subroutine_dispatch(cls.forward)

    @classmethod
    def _subroutine_dispatch(cls, old_forward):
        @functools.wraps(old_forward)
        def new_forward(self, *args, **kwargs):
            if not self.define_subroutine:
                return old_forward(self, *args, **kwargs)

            block_builder = relax.BlockBuilder.current()
            assert block_builder is not None, (
                f"Class {type(self)} has cls.define_subroutines = True, "
                "but is called outsdie of a block_builder environment.  "
                "relax.BlockBuilder.current() is required "
                "to determine where to generate the subroutine."
            )

            func_args = self._normalize_subroutine_args(block_builder, *args, **kwargs)
            subroutine, is_nn_tensor_output = self._get_subroutine(
                block_builder, old_forward, func_args
            )
            subroutine_args = [
                arg._expr if isinstance(arg, nn.Tensor) else arg
                for arg in [*func_args.values(), *self.parameters()]
            ]

            out = subroutine(*subroutine_args)

            if is_nn_tensor_output:
                if isinstance(out.ty, ir.MissingType):
                    out = block_builder.emit(out, name_hint=f"{subroutine.name_hint}_output")
                out = nn.Tensor(_expr=out)
            return out

        new_forward._is_subroutine_mixin = True

        return new_forward

    def _normalize_subroutine_args(
        self, block_builder, *args, **kwargs
    ) -> typing.OrderedDict[str, relax.Expr]:
        signature = inspect.signature(self.forward)
        bindings = signature.bind(*args, **kwargs)
        func_args = collections.OrderedDict(
            (name, _normalize_expr(block_builder, arg)) for name, arg in bindings.arguments.items()
        )
        return func_args

    def _get_subroutine(
        self,
        block_builder,
        old_forward: typing.Callable,
        func_args: typing.OrderedDict[str, relax.Expr],
    ) -> (ir.GlobalVar, bool):
        cls = type(self)
        if not hasattr(cls, "_gvar"):
            cls._gvar = {}

        model_params = [
            param._expr if isinstance(param, nn.Tensor) else param for param in self.parameters()
        ]

        arg_ty = _get_ty([*func_args.values(), *model_params])
        is_dataflow = block_builder.current_block_is_dataflow()
        lookup_key = (
            old_forward,
            tvm_ffi.structural_hash(arg_ty, map_free_vars=True),
            is_dataflow,
        )

        for cached_ty, cached_result in cls._gvar.get(lookup_key, []):
            if tvm_ffi.structural_equal(cached_ty, arg_ty, map_free_vars=True):
                return cached_result

        func_name = _camel_to_snake(cls.__name__)
        func_params = [relax.Var(name, ty) for name, ty in zip(func_args, arg_ty.fields)]
        old_forward_args = [
            nn.Tensor(_expr=param) if isinstance(old_arg, nn.Tensor) else param
            for param, old_arg in zip(func_params, func_args.values())
        ]

        with block_builder.function(func_name, [*func_params, *model_params], private=True):
            with contextlib.ExitStack() as stack:
                if is_dataflow:
                    stack.enter_context(block_builder.dataflow())

                out = old_forward(self, *old_forward_args)
                is_nn_tensor_output = isinstance(out, nn.Tensor)
                if is_nn_tensor_output:
                    out = out._expr

                if is_dataflow:
                    out = block_builder.emit_output(out)
            gvar = block_builder.emit_func_output(out)

        # The relax.Var instances in model_params, along with any
        # tirx.Var instances in the type, appear in both the
        # calling scope and as parameters for the subroutine.  To
        # maintain SSA, replace all relax and TIR variables in the
        # subroutine.
        mod = block_builder.get()
        mod.update_func(gvar, relax.utils.copy_with_new_vars(mod[gvar]))

        result = (gvar, is_nn_tensor_output)
        bucket = cls._gvar.setdefault(lookup_key, [])
        bucket.append((arg_ty, result))
        return result
