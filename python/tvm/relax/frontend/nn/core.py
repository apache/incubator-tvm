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
"""The core infra for nn.Module, which includes the following pieces:
- Tensor, a wrapper on top of relax.Expr whose ty is a TensorType,
  providing more convenient access shape and dtype information.
  Tensor is always symbolic and not bound to any concrete values.
- Parameter, a special tensor which could be bound or not bound to concrete values.
- Module, a container of nn.Parameters and sub nn.Modules.
- Effect, a non-user-facing class that encloses potential side effects, for example, IO,
  impure external function callings, inplace mutation, etc.
"""

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

import numpy as np  # type: ignore
import tvm_ffi

import tvm_tirx.runtime
from tvm.runtime.vm import VirtualMachine
from tvm_tirx import tirx
from tvm_tirx.ir import IRModule
from tvm_tirx.ir.transform import Pass
from tvm_tirx.runtime import Device
from tvm_tirx.target import Target

from .... import relax as rx
from ...block_builder import BlockBuilder
from ...type import (
    AnyType,
    ShapeType,
    TensorType,
    TupleType,
)
from ._tensor_op import _TensorOp
from .subroutine import SubroutineMixin

if TYPE_CHECKING:
    import torch  # type: ignore

    from . import spec as _spec
    from .extern import ExternModule


_DEFAULT_DTYPE = "float32"


def get_default_dtype() -> str:
    """Get the default parameter dtype if not specified. By default it is float32.

    Returns
    -------
    dtype : str
        The default dtype
    """
    return _DEFAULT_DTYPE


def set_default_dtype(dtype: str) -> None:
    """Set the default parameter dtype.

    Parameters
    ----------
    dtype : str
        The default dtype to be set
    """
    global _DEFAULT_DTYPE  # pylint: disable=global-statement
    _DEFAULT_DTYPE = dtype


class Tensor(_TensorOp):
    """A wrapper on top of relax.Expr whose ty is a TensorType, providing more
    convenient access shape and dtype information. Tensor is always symbolc and not bound to any
    concrete values. Shape and dtype inference is done eagerly upon tensor creation, i.e. when
    operators are applied on tensors, the shape and dtype information is already available.
    """

    _expr: rx.Expr

    def __init__(self, *, _expr: rx.Expr) -> None:
        """Private constructor. Tensor is never supposed to be constructed directly by users."""

        def _check_tensor(expr: rx.Expr) -> None:
            assert expr.ty is not None
            assert isinstance(expr.ty, TensorType)
            assert expr.ty.ndim != -1
            assert expr.ty.shape is not None
            assert expr.ty.shape.ty is not None
            assert isinstance(expr.ty.shape.ty, ShapeType)
            assert expr.ty.shape.ty.values is not None

        _check_tensor(_expr)
        self._expr = _expr

    @staticmethod
    def from_const(data) -> "Tensor":
        """Construct a tensor from numpy constants."""
        return Tensor(_expr=rx.const(data))

    @staticmethod
    def from_scalar(data: int | float, dtype: str) -> "Tensor":
        """Construct a tensor from a scalar with dtype specified."""
        return Tensor(_expr=rx.const(data, dtype=dtype))

    @staticmethod
    def from_ty(ty: rx.TensorType, name: str = "tensor") -> "Tensor":
        """Construct a nn.Tensor from a Relax TensorType.

        TensorType is the Relax type-level description of a tensor, carrying its shape
        and dtype without holding actual data. This factory creates an unbound placeholder
        ``nn.Tensor`` that can be used as a symbolic input when tracing an ``nn.Module``.

        Parameters
        ----------
        ty : rx.TensorType
            The type describing the tensor's shape and dtype.

        name : str
            Name hint for the underlying Relax variable.

        Returns
        -------
        tensor : Tensor
            A symbolic ``nn.Tensor`` backed by a ``relax.Var`` with the given type.
        """
        return Tensor(
            _expr=rx.Var(
                name=name,
                ty=ty,
            )
        )

    @staticmethod
    def placeholder(
        shape: Sequence[int | str | tirx.Expr],
        dtype: str,
        name: str = "tensor",
    ) -> "Tensor":
        """Create a placeholder tensor with given shape and dtype. A placeholder tensor should
        never be created directly by users in usual cases, and the only exception is to indicate
        the shape/dtype of return values of an external function.

        If shape is a string `name`, we create a symbolic shape `tvm_tirx.tirx.Var(name, "int64")`.
        """
        new_shape = []
        for expr in shape:
            if isinstance(expr, int | tvm_tirx.ir.IntImm):
                expr = int(expr)
                assert expr >= 0
                new_shape.append(expr)
                continue
            if isinstance(expr, str):
                expr = tirx.Var(expr, "int64")
                new_shape.append(expr)
                continue
            if not tvm_tirx.ir.is_prim_expr(expr):
                raise TypeError(f"Invalid shape: {shape}")
            assert expr.ty == tvm_tirx.ir.PrimType("int64")
            new_shape.append(expr)
        return Tensor(
            _expr=rx.Var(
                name=name,
                ty=TensorType(
                    shape=new_shape,  # type: ignore[arg-type]
                    dtype=dtype,
                ),
            )
        )

    @property
    def shape(self) -> list[int | tirx.Expr]:
        """Returns the shape of the tensor as a list of integers.

        An integer can be a python int or tvm_tirx.tirx.Expr, depending on whether the shape is
        fully static, for example, [1, 2, tvm_tirx.tirx.Var("n", "int64")] is a valid shape where
        the last dimension is dynamic while the first two dimensions are always static constants.

        Returns
        -------
        shape : List[Union[int, tirx.Expr]]
            The shape of the tensor
        """

        def _simplify(expr: tirx.Expr):
            return expr.value if isinstance(expr, tvm_tirx.ir.IntImm) else expr

        shape_ty: ShapeType = self._expr.ty.shape.ty
        return [_simplify(x) for x in shape_ty.values]

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor.

        Returns
        -------
        ndim : int
            The number of dimensions of the tensor
        """
        return self._expr.ty.ndim

    @property
    def dtype(self) -> str:
        """Returns the data type of the tensor.

        Returns
        -------
        dtype : str
            The data type of the tensor
        """
        return self._expr.ty.dtype.dtype

    def __repr__(self) -> str:
        return f'Tensor({self.shape}, "{self.dtype}")'


class Parameter(Tensor):
    """A parameter represents the weight of a neural network layer. It is a special tensor which
    could be bound or not bound to concrete values. If a parameter is bound to a concrete value,
    it is called a bound parameter, otherwise it is called an unbound parameter.
    """

    _data: Tensor | None
    attrs: dict[str, Any]

    def __init__(
        self,
        shape: Sequence[int | str | tirx.Expr],
        dtype: str | None = None,
    ) -> None:
        """Create a parameter with given shape and dtype. The parameter is not bound to any
        concrete values.

        Parameters
        ----------
        shape : Sequence[Union[int, str, tirx.Expr]]
            The shape of the parameter. If it is a string `name`, we create a symbolic shape
            `tvm_tirx.tirx.Var(name, "int64")`.
        dtype : Optional[str]
            The data type of the parameter. If not specified, the default dtype will be used.
        """
        if dtype is None:
            dtype = get_default_dtype()
        super().__init__(_expr=Tensor.placeholder(shape, dtype=dtype, name="param")._expr)
        self._data = None
        self.attrs = OrderedDict()

    @property
    def data(self) -> Tensor | None:
        """Returns the concrete value of the parameter if it is bound to a concrete value,
        otherwise returns None. The returned value is a tvm_tirx.runtime.Tensor."""
        return self._data

    @data.setter
    def data(self, data: Union[None, tvm_tirx.runtime.Tensor, np.ndarray, "torch.Tensor"]) -> None:
        """Set the concrete value of the parameter. The data should be one of the following:
        - None: unbind the parameter to concrete values
        - tvm_tirx.runtime.Tensor
        - numpy.ndarray
        - torch.Tensor and any other DLPack-compliant tensors
        """
        if data is None:
            self._data = data
            return
        # Try to do zero-copy if possible
        if isinstance(data, tvm_tirx.runtime.Tensor):
            pass
        elif isinstance(data, np.ndarray):
            data = tvm_tirx.runtime.tensor(data)
        elif hasattr(data, "__dlpack__"):
            data = _from_dlpack(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        if data.shape != tuple(self.shape):
            raise ValueError(f"Shape mismatch: expected {tuple(self.shape)}, got {data.shape}")
        if data.dtype != self.dtype:
            raise ValueError(f"Dtype mismatch: expected {self.dtype}, got {data.dtype}")
        self._data = data

    def to(self, dtype: str | None = None) -> None:  # pylint: disable=invalid-name
        """Change the dtype of the parameter if it is not bound to any concrete data"""
        if dtype is not None:
            if self._data is not None:
                raise ValueError(
                    "Changing the dtype of a Parameter that has been bound to concrete "
                    "data is not recommended. It might lead to potential precision loss "
                    "or other unexpected behaviors"
                )
            self._expr = Tensor.placeholder(  # pylint: disable=protected-access
                self.shape, dtype=dtype, name="param"
            )._expr


class Object:
    """A wrapper on top of relax.Expr whose ty is the base
    AnyType, rather than a more specific subtype. Object effectively
    represents non-tensor frontend components such as KV caches.
    """

    _expr: rx.Var

    def __init__(self, *, _expr: rx.Expr, _name: str) -> None:
        """Private constructor. Object is never supposed to be constructed directly by users."""
        if not isinstance(_expr, rx.Var):
            _expr = BlockBuilder.current().emit(_expr, _name)
        self._expr = _expr
        assert isinstance(self._expr.ty, AnyType)


class Effect:
    """Effect is a special non-user facing type that is used to represent operations with side
    effects, for example, print. It is used to represent the output of a computation.
    """

    def emit_init(self, name_hint: str, builder: BlockBuilder) -> list[rx.DataflowVar]:
        """Emit the initialization of the effect. This method is called by the compiler to
        initialize the effect."""
        raise NotImplementedError

    def create(self, name_hint: str) -> list[rx.Var]:
        """Create the implicit inputs to a relax.Function that represents the side effect"""
        raise NotImplementedError

    def set_state(self, state_vars: list[rx.Var]) -> None:
        """Set the variables that represents the effect"""
        raise NotImplementedError

    def finalize(self) -> list[rx.Var]:
        """finalize the effect as the implicit return value of a relax.Function"""
        raise NotImplementedError

    def to(self, dtype: str | None = None) -> None:  # pylint: disable=invalid-name
        """Convert the effect to specific dtype. Usually it is no-op for most of the effects"""


class Module(SubroutineMixin):
    """Base class for neural network components. Subclass it to build your models.
    Modules can nest within each other in a tree structure using regular attribute assignment."""

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """This method provides an iterator over module parameters,
        yielding both the parameter name and its corresponding value.

        Parameters
        ----------
        prefix : str
            Prefix to prepend to all parameter names.

        Yields
        ------
        (str, Parameter) - Tuple containing the name and parameter
        """
        yield from _attribute_finder(
            self, prefix, condition_yield=lambda x: isinstance(x, Parameter)
        )

    def parameters(self) -> Iterator[Parameter]:
        """This method provides an iterator over module parameters,
        yielding only the Parameter value.

        Yields
        ------
        Parameter - The module's parameter
        """
        for _, param in self.named_parameters():
            yield param

    def state_dict(
        self, *, prefix: str = "", destination: dict[str, Parameter] | None = None
    ) -> dict[str, Parameter]:
        """Returns a dictionary containing references to the whole state of the module.

        Parameters
        ----------
        prefix : str
            Prefix to prepend to all parameter names.
        destination : Optional[Dict[str, Parameter]]
            Dictionary to which state will be saved. If None, a new dictionary is created.

        Returns
        -------
        dict : Dict[str, Parameter]
            a dictionary containing a whole state of the module
        """
        if destination is None:
            destination = OrderedDict()
        for name, param in _attribute_finder(
            self, prefix, condition_yield=lambda x: isinstance(x, Parameter)
        ):
            destination[name] = param
        return destination

    def load_state_dict(
        self, state_dict: dict[str, Parameter], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        """This function copies parameters and buffers from the state_dict into the current module
        and its descendants. If `strict` is set to True, the keys in the `state_dict` must exactly
        match the keys returned by the `state_dict()` function of this module.

        Parameters
        ----------
        state_dict : Dict[str, Parameter]
            A dictionary containing a whole state of the module
        strict : bool = True
            Whether to strictly enforce that the keys in `state_dict` match the keys returned by
            this module's `state_dict()` function.

        Returns
        -------
        (missing_keys, unexpected_keys) : Tuple[List[str], List[str]]
            A tuple of two lists: the missing keys and the unexpected keys.
        """
        self_state_dict = self.state_dict()
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        for key, value in state_dict.items():
            if key not in self_state_dict:
                unexpected_keys.append(key)
                continue
            if value.data is None:
                raise ValueError(f"Parameter {key} is not set to any concrete tensor")
            self_state_dict.pop(key).data = value.data
        missing_keys = list(self_state_dict.keys())
        if strict and (missing_keys or unexpected_keys):
            raise KeyError(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        return missing_keys, unexpected_keys

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the module with the given inputs and returns the output."""
        if not hasattr(self, "forward"):
            raise NotImplementedError(f"Module {type(self)} does not have a `forward` method")
        return self.forward(*args, **kwargs)  # pylint: disable=no-member

    def to(self, dtype: str | None = None) -> None:  # pylint: disable=invalid-name
        """Convert the module to specific dtype recursively"""
        for _, item in self.__dict__.items():
            if hasattr(item, "to") and callable(item.to):
                item.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init

    def export_tvm(
        self,
        spec: "_spec.ModuleSpecType",
        debug: bool = False,
        allow_extern: bool = False,
    ) -> (
        tuple[IRModule, list[tuple[str, Parameter]]]
        | tuple[IRModule, list[tuple[str, Parameter]], list["ExternModule"]]
    ):
        """Export the module to TVM IRModule and parameters

        Parameters
        ----------
        spec : _spec.ModuleSpecType
            A dictionary mapping each input name to a specification
            that defines the inputs shape and dtype.
        debug : bool
            If set to True, then the exported module will support
            effects. This enables things like printing in the graph.

        Returns
        -------
        irmodule : tvm_tirx.ir.IRModule
            The converted tvm IR representation of the model.
        params : List[Tuple[str, Parameter]]
            A list of Parameters corresponding to the weights of the model.
        ext_mods : List[nn.ExternModule]
            A list of ExternModules that are used in the model.
        """
        # pylint: disable=import-outside-toplevel
        from . import spec as _spec
        from .exporter import Exporter

        # pylint: enable=import-outside-toplevel
        spec = _spec.ModuleSpec.from_raw(spec, self)
        mod, params, ext_mods = Exporter(debug=debug).build(spec)
        if allow_extern:
            return mod, params, ext_mods
        if ext_mods:
            raise ValueError(
                "`ExternModule`(s) exist when they are not allowed. "
                "Turn on flag `allow_extern` to allow."
            )
        return mod, params

    def jit(  # pylint: disable=too-many-arguments
        self,
        spec: "_spec.ModuleSpec",
        device: str | Device = "cpu",
        pipeline: None | str | Pass = "default_build",
        out_format: str = "torch",
        debug: bool = False,
    ) -> Any:
        """Just-in-time compile an ``nn.Module`` into a callable executable.

        The method exports the module to a Relax IRModule, applies the given compilation
        pipeline, builds a Relax VM executable, and wraps the result so it can be called
        directly (e.g. with PyTorch tensors when ``out_format="torch"``).

        Parameters
        ----------
        spec : _spec.ModuleSpec
            A specification mapping each module input to its shape and dtype.

        device : Union[str, Device]
            The device to compile and run on (e.g. ``"cpu"``, ``"cuda"``).

        pipeline : Union[None, str, Pass]
            The Relax compilation pipeline to apply. ``"default_build"`` uses the standard
            optimization pipeline; ``None`` skips pipeline passes.

        out_format : str
            Output wrapper format. ``"torch"`` returns a ``TorchModule`` whose ``forward``
            accepts and returns PyTorch tensors.

        debug : bool
            If ``True``, enable effect-based debugging (e.g. printing) in the compiled graph.

        Returns
        -------
        module : Any
            A callable wrapper (type depends on *out_format*) around the compiled VM.
        """

        def _compile(spec, device, pipeline, debug):
            # pylint: disable=import-outside-toplevel
            from ...transform import AttachExternModules
            from ...vm_build import build as relax_build
            from . import spec as _spec
            from .exporter import Exporter

            # pylint: enable=import-outside-toplevel

            spec = _spec.ModuleSpec.from_raw(spec, self)
            mod, params, ext_mods = Exporter(debug=debug).build(spec)
            mod = AttachExternModules(ext_mods)(mod)  # pylint: disable=not-callable
            vm = VirtualMachine(  # pylint: disable=invalid-name
                relax_build(
                    mod,
                    target=Target.from_device(device),
                    relax_pipeline=pipeline,
                ),
                device,
            )
            params = _param_to_tensor(params, device)
            return spec, vm, params

        device = tvm_ffi.device(device)
        spec, vm, params = _compile(spec, device, pipeline, debug)  # pylint: disable=invalid-name

        if out_format == "torch":
            from . import torch  # pylint: disable=import-outside-toplevel

            return torch.TorchModule(spec=spec, params=params, vm=vm)

        raise ValueError(f"Unknown out_format: {out_format}")


class ModuleDict(Module):
    """Holds submodules in a dict."""

    def __init__(self, modules: OrderedDict[str, Module] | None = None):
        if modules is None:
            self.modules = OrderedDict()
        else:
            self.modules = OrderedDict(modules)

    def __iter__(self):
        return iter(self.modules.values())

    def __getitem__(self, key: str) -> Module:
        return self.modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.modules[key] = module

    def __len__(self) -> int:
        return len(self.modules)

    def keys(self) -> Iterator[str]:
        return self.modules.keys()

    def values(self) -> Iterator[Module]:
        return self.modules.values()

    def items(self) -> Iterator[tuple[str, Module]]:
        return self.modules.items()

    def get(self, key: str, default: Module | None = None) -> Module | None:
        return self.modules.get(key, default)

    def update(self, modules: dict[str, Module]) -> None:
        self.modules.update(modules)

    def clear(self) -> None:
        self.modules.clear()

    def pop(self, key: str) -> Module:
        return self.modules.pop(key)

    def __contains__(self, key: str) -> bool:
        return key in self.modules

    def to(self, dtype: str | None = None) -> None:  # pylint: disable=invalid-name
        for module in self.modules.values():
            module.to(dtype=dtype)


class ParameterDict(Module):
    """Holds parameters in a dict."""

    def __init__(
        self,
        params: OrderedDict[str, Parameter] | dict[str, Parameter] | None = None,
    ):
        self.params: OrderedDict[str, Parameter] = OrderedDict()
        if params is not None:
            self.update(params)

    def __iter__(self) -> Iterator[str]:
        return iter(self.params)

    def __getitem__(self, key: str) -> Parameter:
        return self.params[key]

    def __setitem__(self, key: str, param: Parameter) -> None:
        if not isinstance(key, str):
            raise TypeError(f"ParameterDict keys must be strings, but got {type(key).__name__}")
        if not isinstance(param, Parameter):
            raise TypeError(
                f"ParameterDict values must be nn.Parameter, but got {type(param).__name__}"
            )
        self.params[key] = param

    def __len__(self) -> int:
        return len(self.params)

    def keys(self) -> Iterator[str]:
        return self.params.keys()

    def values(self) -> Iterator[Parameter]:
        return self.params.values()

    def items(self) -> Iterator[tuple[str, Parameter]]:
        return self.params.items()

    def get(self, key: str, default: Parameter | None = None) -> Parameter | None:
        return self.params.get(key, default)

    def update(self, params: dict[str, Parameter]) -> None:
        for key, param in params.items():
            self[key] = param

    def clear(self) -> None:
        self.params.clear()

    def pop(self, key: str) -> Parameter:
        return self.params.pop(key)

    def __contains__(self, key: str) -> bool:
        return key in self.params

    def to(self, dtype: str | None = None) -> None:  # pylint: disable=invalid-name
        for param in self.params.values():
            param.to(dtype=dtype)


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules: list[Module]):
        self.modules = modules

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx: int) -> Module:
        return self.modules[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        self.modules[idx] = module

    def __len__(self):
        return len(self.modules)

    def append(self, module: Module):
        """Add a module to the end of the ModuleList"""
        self.modules.append(module)

    def to(self, dtype: str | None = None) -> None:  # pylint: disable=invalid-name
        for module in self.modules:
            module.to(dtype=dtype)

    def forward(self, x):  # pylint: disable=invalid-name
        """Feed-forward pass of the module"""
        for module in self.modules:
            x = module(x)
        return x


class ParameterList(Module):
    """Holds parameters in a list."""

    def __init__(self, params: list[Parameter] | None = None):
        self.params: list[Parameter] = []
        if params is not None:
            self.extend(params)

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self.params)

    def __getitem__(self, idx: int) -> Parameter:
        return self.params[idx]

    def __setitem__(self, idx: int, param: Parameter) -> None:
        if not isinstance(param, Parameter):
            raise TypeError(
                f"ParameterList elements must be nn.Parameter, but got {type(param).__name__}"
            )
        self.params[idx] = param

    def __len__(self) -> int:
        return len(self.params)

    def append(self, param: Parameter) -> None:
        """Add a parameter to the end of the ParameterList"""
        if not isinstance(param, Parameter):
            raise TypeError(
                f"ParameterList elements must be nn.Parameter, but got {type(param).__name__}"
            )
        self.params.append(param)

    def extend(self, params: list[Parameter]) -> None:
        """Add parameters to the end of the ParameterList"""
        for param in params:
            self.append(param)

    def to(self, dtype: str | None = None) -> None:  # pylint: disable=invalid-name
        for param in self.params:
            param.to(dtype=dtype)


def wrap_nested(expr: rx.Expr, name: str) -> Tensor | Sequence[Tensor]:
    """Wrap the given relax.Expr, emit it using the current BlockBuilder,
    and automatically handle nested cases if the expr represents a Tuple.

    Parameters
    ----------
    expr : relax.Expr
        The Expr to be wrapped.

    name : str
        Name hint.

    Returns
    -------
    result : Union[Tensor, Tuple[Tensor]]
        The computed result.
    """
    if not isinstance(expr, rx.DataflowVar):
        expr = BlockBuilder.current().emit(expr, name)
    if isinstance(expr.ty, TensorType):
        return Tensor(_expr=expr)
    if isinstance(expr.ty, TupleType):
        return tuple(
            wrap_nested(  # type: ignore
                rx.TupleGetItem(expr, i),
                name=f"{name}.{i}",
            )
            for i in range(len(expr.ty.fields))
        )
    raise TypeError(f"Unsupported return type: {expr.ty}")


def _attribute_finder(root: Module, prefix: str, condition_yield: Callable[[Any], bool]):
    """Find attributes that satisfy the condition recursively"""
    if isinstance(root, ParameterList):
        for i, param in enumerate(root):
            if condition_yield(param):
                yield prefix + f"{i}", param
        return
    elif isinstance(root, ParameterDict):
        for name, param in root.items():
            if condition_yield(param):
                yield prefix + name, param
        return
    elif isinstance(root, ModuleList):
        for i, subitem in enumerate(root):
            yield from _attribute_finder(subitem, prefix + f"{i}.", condition_yield)
        return
    elif isinstance(root, ModuleDict):
        for name, subitem in root.items():
            yield from _attribute_finder(subitem, prefix + f"{name}.", condition_yield)
        return
    for name, item in root.__dict__.items():
        if condition_yield(item):
            yield prefix + name, item
        elif isinstance(item, ParameterList):
            yield from _attribute_finder(
                item,
                prefix + name + ".",
                condition_yield,
            )
        elif isinstance(item, ParameterDict):
            yield from _attribute_finder(
                item,
                prefix + name + ".",
                condition_yield,
            )
        elif isinstance(item, ModuleList):
            yield from _attribute_finder(
                item,
                prefix + name + ".",
                condition_yield,
            )
        elif isinstance(item, ModuleDict):
            for sub_name, sub_item in item.items():
                yield from _attribute_finder(
                    sub_item,
                    prefix + name + f".{sub_name}.",
                    condition_yield,
                )
        elif isinstance(item, Module):
            yield from _attribute_finder(
                item,
                prefix + name + ".",
                condition_yield,
            )


def _from_dlpack(tensor) -> tvm_tirx.runtime.Tensor:
    try:
        return tvm_ffi.from_dlpack(tensor)
    except RuntimeError:
        pass
    # special logic for PyTorch
    device_type = tensor.device.type
    device_id = tensor.device.index or 0
    return tvm_tirx.runtime.tensor(
        tensor.numpy(),
        device=Device(
            Device._DEVICE_NAME_TO_TYPE[device_type],
            device_id,
        ),
    )


def _param_to_tensor(
    params: list[tuple[str, Parameter]], device: Device
) -> list[tvm_tirx.runtime.Tensor]:
    results = []
    missing = []
    for name, param in params:
        if param.data is None:
            missing.append(name)
        else:
            results.append(param.data.copyto(target=device))
    if missing:
        raise ValueError(f"Parameters are not set to any concrete values: {', '.join(missing)}")
    return results
