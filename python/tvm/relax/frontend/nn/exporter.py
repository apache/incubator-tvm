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
"""Export `nn.Module` to TVM's IRModule."""

import functools
import operator
import threading
import typing

import tvm_tirx
from tvm_tirx import tirx
from tvm_tirx.ir import IRModule

from .... import relax as rx
from ...block_builder import BlockBuilder
from ...type import AnyType, ShapeType, TupleType
from . import core, extern
from . import spec as _spec
from .modules import IOEffect


def add_extern(mod: extern.ExternModule) -> None:
    """Add an external module to the exporter."""
    try:
        exporter = Exporter.current()
    except Exception as exception:
        raise RuntimeError(
            "`nn.add_extern` should only be invoked when exporting a module."
        ) from exception
    exporter.add_external_module(mod)


class Exporter:
    """Builder of ModuleSpec, which exports an nn.Module to TVM IRModule."""

    _tls = threading.local()

    builder: BlockBuilder
    io_effect: core.Effect
    extern_mods: list[extern.ExternModule]

    def __init__(self, debug: bool) -> None:
        self.builder = BlockBuilder()
        self.io_effect = IOEffect() if debug else None
        self.extern_mods = []

    @staticmethod
    def current() -> "Exporter":
        """Get the current Exporter under the with scope."""
        assert hasattr(Exporter._tls, "current")
        return Exporter._tls.current

    def __enter__(self) -> "Exporter":
        assert not hasattr(Exporter._tls, "current")
        Exporter._tls.current = self
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert hasattr(Exporter._tls, "current")
        delattr(Exporter._tls, "current")

    def add_external_module(self, mod: extern.ExternModule) -> None:
        """Add an external module to the exporter."""
        # pylint: disable=protected-access
        all_symbols: list[str] = []
        for extern_mod in self.extern_mods:
            all_symbols.extend(extern_mod._symbols.keys())
        duplicated_symbols = list(set(mod._symbols.keys()) & set(all_symbols))
        # pylint: enable=protected-access
        if duplicated_symbols:
            raise ValueError(f"Duplicate symbols: {duplicated_symbols}")
        self.extern_mods.append(mod)

    def build(  # pylint: disable=too-many-locals
        self,
        spec: _spec.ModuleSpec,
    ) -> tuple[
        IRModule,
        list[tuple[str, core.Parameter]],
        list[extern.ExternModule],
    ]:
        """Build the ModuleSpec to TVM IRModule. Returns the IRModule and the parameters."""

        # pylint: disable=protected-access
        def _params() -> list[tuple[str, core.Parameter]]:
            params = []
            for name, param in core._attribute_finder(
                spec.module, prefix="", condition_yield=lambda x: isinstance(x, core.Parameter)
            ):
                params.append((name, param))
            return params

        def _effects() -> list[tuple[str, core.Effect]]:
            result = []
            if self.io_effect is not None:
                result.append(("", self.io_effect))
            for name, effect in core._attribute_finder(
                spec.module, "", condition_yield=lambda x: isinstance(x, core.Effect)
            ):
                result.append((name, effect))
            return result

        # pylint: enable=protected-access
        params = None
        effects = _effects()
        ext_mods = self.extern_mods
        with self:
            if effects:
                with self.builder.function("_initialize_effect"):
                    with self.builder.dataflow():
                        outputs = _emit_effect_init(self.builder, effects)
                    self.builder.emit_func_output(outputs, params=[])
            for method_name, method_spec in zip(spec.method_names, spec.method_specs):
                params = _params()  # Re-initialize so symbolic shapes not shared across methods
                len_args = len(method_spec.arg_specs)
                len_effects = {
                    "packed": 1,
                    "none": 0,
                    "plain": len(effects),
                }[method_spec.effect_mode]
                with self.builder.function(
                    method_name,
                    attrs={"num_input": len_args + len_effects},  # type: ignore
                ):
                    with self.builder.dataflow():
                        outputs, inputs = _emit_method(self.builder, method_spec, params, effects)
                    self.builder.emit_func_output(outputs, inputs)
        mod = self.builder.finalize()
        rx.analysis.check_well_formed(mod)

        return mod, params, ext_mods


def _emit_effect_init(
    builder: BlockBuilder,
    effects: list[tuple[str, core.Effect]],
):
    outputs = []
    for prefix, effect in effects:
        inits = effect.emit_init(prefix, builder)
        assert isinstance(inits, list)
        outputs.extend(inits)
    outputs = builder.emit_output(builder.emit(rx.Tuple(outputs)))
    return outputs


def _emit_method(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    builder: BlockBuilder,
    spec: _spec.MethodSpec,
    params: list[tuple[str, core.Parameter]],
    effects: list[tuple[str, core.Effect]] | None,
):
    # pylint: disable=protected-access
    # symbolic shape's name mapping to its tirx.Var for reuse
    str2var_params: dict[str, tirx.Var] = {}

    def _unwrap_ret(expr: typing.Any) -> typing.Any:
        if isinstance(expr, core.Tensor | core.Object):
            return expr._expr
        if isinstance(expr, tuple):
            return rx.Tuple([_unwrap_ret(x) for x in expr])
        if isinstance(expr, list):
            return rx.Tuple([_unwrap_ret(x) for x in expr])
        raise TypeError(f"Unsupported return type: {type(expr)}")

    def _convert_input(arg):
        if isinstance(arg, tvm_tirx.ir.Var):
            return rx.Var(arg.name, ty=ShapeType(values=[arg]))
        if isinstance(arg, core.Tensor | core.Object):
            return arg._expr  # pylint: disable=protected-access
        if isinstance(arg, _spec.Tuple):
            return rx.Var(
                arg.name,
                ty=TupleType([_convert_input(arg_i).ty for arg_i in arg.elements]),
            )
        raise TypeError(f"Unsupported input type: {type(arg)}")

    def _params(mode: str) -> list[rx.Var]:
        inputs: list[rx.Var] = []

        def _get_var(shape_var: tirx.Var) -> tirx.Var:
            name = shape_var.name
            if name in str2var_params:
                return str2var_params[name]
            var = tirx.Var(name, "int64")
            str2var_params[name] = var
            return var

        for name, param in params:
            # Make sure the a symbolic shape is not re-registered (same as _method_spec_to_inputs)
            # e.g. we do not see `vocab_size` for `lm_head` and `vocab_size_1` for `embed_tokens`
            new_shape = [_get_var(x) if isinstance(x, tvm_tirx.ir.Var) else x for x in param.shape]
            var = core.Tensor.placeholder(new_shape, param.dtype, name)._expr
            inputs.append(var)
            param._expr = var
        if mode == "none":
            return []
        if mode == "plain":
            return inputs
        if mode == "packed":
            input_var = rx.Var(
                "packed_params",
                TupleType(fields=[x.ty for x in inputs]),
            )
            for i, (name, param) in enumerate(params):
                param._expr = builder.emit(rx.TupleGetItem(input_var, i), name_hint=name)
            return [input_var]
        raise ValueError(f"Invalid param_mode: {mode}")

    def _effects(mode: str) -> list[rx.Var]:
        unflat_inputs: list[list[rx.Var]] = []
        for name, effect in effects:
            effect_input = effect.create(name)
            effect.set_state(effect_input)
            unflat_inputs.append(effect_input)
        inputs: list[rx.Var] = functools.reduce(operator.iadd, unflat_inputs, [])
        if mode == "none":
            return []
        if mode == "plain":
            return inputs
        if mode == "packed":
            input_var = rx.Var(
                "packed_effects",
                TupleType(fields=[x.ty for x in inputs]),
            )
            i = 0
            for effect_input, (_, effect) in zip(unflat_inputs, effects):
                updated_effect_input = []
                for effect_input_i in effect_input:
                    updated_effect_input.append(
                        builder.emit(
                            rx.TupleGetItem(input_var, i),
                            name_hint=effect_input_i.name,
                        )
                    )
                    i += 1
                effect.set_state(updated_effect_input)
            return [input_var]

        raise ValueError(f"Invalid effect_mode: {mode}")

    # pylint: enable=protected-access

    def _detuple(arg, var: rx.Var, builder: BlockBuilder):
        if isinstance(arg, _spec.Tuple):
            ret = []
            for i, elem in enumerate(arg.elements):
                field = builder.emit(rx.TupleGetItem(var, i), name_hint=f"{arg.name}_{i}")
                ret.append(_detuple(elem, field, builder))
            return type(arg.elements)(ret)
        if isinstance(arg, core.Tensor):
            return core.Tensor(_expr=var)
        if isinstance(arg, tvm_tirx.ir.Var):
            return arg
        raise TypeError(f"Unsupported input type: {type(arg)}")

    # TODO(@junrushao): Warn if params/effects are used when their mode is "none"
    explicit_inputs = _method_spec_to_inputs(spec)
    inputs = [_convert_input(x) for x in explicit_inputs]
    inputs = inputs + _effects(spec.effect_mode)
    inputs = inputs + _params(spec.param_mode)

    for arg_idx, (arg, var) in enumerate(zip(explicit_inputs, inputs)):
        if isinstance(arg, _spec.Tuple):
            explicit_inputs[arg_idx] = _detuple(arg, var, builder)

    outputs = spec.method(*explicit_inputs)
    effect_outputs = []
    for _, effect in effects:
        effect_outputs.extend(effect.finalize())
    if effect_outputs and spec.effect_mode != "none":
        outputs = builder.emit_output(rx.Tuple([_unwrap_ret(outputs), rx.Tuple(effect_outputs)]))
    else:
        outputs = builder.emit_output(_unwrap_ret(outputs))
    return outputs, inputs


def _method_spec_to_inputs(
    spec: _spec.MethodSpec,
) -> list[tirx.Var | core.Tensor]:
    """Convert the MethodSpec to a list of inputs to Module's method."""
    str2var: dict[str, tirx.Var] = {}

    def _get_var(name: str) -> tirx.Var:
        if name in str2var:
            return str2var[name]
        var = tirx.Var(name, "int64")
        str2var[name] = var
        return var

    def _convert_input(arg_name, arg_spec):
        if isinstance(arg_spec, _spec.Int):
            arg = _get_var(arg_name)
        elif isinstance(arg_spec, _spec.Tensor):
            arg = core.Tensor.placeholder(  # pylint: disable=protected-access
                shape=[_get_var(x) if isinstance(x, str) else x for x in arg_spec.shape],
                dtype=arg_spec.dtype,
                name=arg_name,
            )
        elif isinstance(arg_spec, _spec.Object):
            arg = arg_spec.object_type(_expr=rx.Var(arg_name, AnyType()), _name=arg_name)
        elif isinstance(arg_spec, _spec.Tuple):
            elements = type(arg_spec.elements)(
                [
                    _convert_input(arg_name=arg_name + f"_{i}", arg_spec=arg_spec.elements[i])
                    for i in range(len(arg_spec.elements))
                ]
            )
            arg = _spec.Tuple(
                name=arg_name,
                elements=elements,
            )
        else:
            raise TypeError(f"Invalid spec for argument {arg_name}: {arg_spec}")
        return arg

    args = []
    for arg_name, arg_spec in zip(spec.arg_names, spec.arg_specs):
        arg = _convert_input(arg_name=arg_name, arg_spec=arg_spec)
        args.append(arg)
    return args
