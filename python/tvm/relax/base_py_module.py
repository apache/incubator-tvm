# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ruff: noqa: F401, F821
"""BasePyModule: Base class for IRModules with Python function support."""

import inspect
import sys
import weakref
from typing import Any, Optional, Union

import numpy as np
import tvm_ffi
from tvm_ffi import Function

import tvm
import tvm_tirx
from tvm import relax
from tvm_tirx import tirx
from tvm_tirx.ir import IRModule
from tvm_tirx.runtime import Device, Tensor
from tvm_tirx.target import Target

try:
    from torch.utils.dlpack import to_dlpack as to_dlpack_legacy
except ImportError:
    to_dlpack_legacy = None

try:
    from tvm_ffi._optional_torch_c_dlpack import load_torch_c_dlpack_extension

    _FASTER_DLPACK_EXTENSION = load_torch_c_dlpack_extension()
except ImportError:
    _FASTER_DLPACK_EXTENSION = None


# Maps a registered Python function name to the id of the BasePyModule that registered it last.
# A module only unregisters names it still owns, so re-registering a name transfers ownership.
_PY_FUNC_OWNERS: dict[str, int] = {}


class BasePyModule:
    """Base class that allows Python functions in IRModule with DLPack conversion.

    This class provides the infrastructure for:
    1. JIT compilation of TIR and Relax functions.
    2. DLPack-based conversion between PyTorch tensors and TVM Tensors.
    3. Wrapping Relax functions for easy Python calling.
    4. Cross-function calls between Python, TIR, and Relax functions.

    Shared IRModules collect Python functions in ``__pyfuncs__``. Decorating a
    subclass with ``R.py_module`` adds this executable runtime interface.
    """

    # Names this instance registered with the VM's Python function registry. The registry is
    # global, so a module must only remove entries it still owns.
    _registered_py_funcs: tuple[str, ...] = ()

    def __del__(self):
        """Unregister the Python functions this module still owns."""
        registered = getattr(self, "_registered_py_funcs", None)
        if not registered:
            return
        try:
            # Release ownership first: if the call below fails, the map must not keep pointing at
            # a destroyed module, whose id() a later module could reuse.
            owned = [name for name in registered if _PY_FUNC_OWNERS.get(name) == id(self)]
            for func_name in owned:
                del _PY_FUNC_OWNERS[func_name]
            # Once finalization starts the registry goes away with the process, and the module
            # globals this needs may already be cleared.
            if not owned or sys.is_finalizing():
                return
            unregister_py_func = tvm_ffi.get_global_func("vm.builtin.unregister_py_func")
            for func_name in owned:
                unregister_py_func(func_name)
        except Exception:  # pylint: disable=broad-except
            # A finalizer must not raise: either the interpreter is shutting down or the runtime
            # is already unusable, and in both cases the registry no longer matters.
            pass

    def __init__(
        self,
        ir_mod: IRModule,
        device: Device,
        target: Target | None = None,
    ):
        """Initialize BasePyModule with JIT compilation and DLPack conversion."""
        self.device = device
        self.ir_mod = ir_mod

        # Delegate IRModule operations
        self.functions = ir_mod.functions
        self.attrs = ir_mod.attrs
        self.global_infos = ir_mod.global_infos
        self.__getitem__ = ir_mod.__getitem__
        self.__setitem__ = ir_mod.__setitem__
        self.functions_items = ir_mod.functions_items
        self.with_attr = ir_mod.with_attr
        self.get_attr = ir_mod.get_attr
        self.update_global_info = ir_mod.update_global_info

        def _getattr_python_function(name: str) -> Any:
            """Support direct attribute access to funcs and IRModule methods."""
            if name in self.pyfuncs:
                return self.pyfuncs[name]
            if name in self.compiled_tir_funcs:
                return self.compiled_tir_funcs[name]
            if self.relax_vm and name in self.relax_func_names:
                try:
                    return self.relax_vm[name]
                except AttributeError:  # More specific exception
                    return None
            if hasattr(self.ir_mod, name):
                return getattr(self.ir_mod, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        self.__getattr__ = _getattr_python_function

        self.compiled_tir_funcs: dict[str, Function] = {}
        self.extern_funcs: dict[str, Function] = {}
        self.tir_func_names: list[str] = []
        self.relax_func_names: list[str] = []
        self.relax_vm: relax.VirtualMachine | None = None
        self.pyfuncs: dict[str, Any] = {}

        if target is None:
            target = Target.from_device(device)
        elif isinstance(target, str):
            target = Target(target)
        self.target = target

        self._collect_function_names()
        self._compile_functions()
        self._wrap_tir_functions()
        self._wrap_relax_functions()
        self._register_python_functions()

    def _collect_function_names(self):
        """Collect names of TIR and Relax functions from IRModule."""
        for global_var, func in self.ir_mod.functions_items():
            if isinstance(func, tirx.Function):
                self.tir_func_names.append(global_var.name_hint)
            elif isinstance(func, relax.Function):
                self.relax_func_names.append(global_var.name_hint)

    def _compile_functions(self):
        """Compile TIR and Relax functions using JIT compilation."""
        # Compile TIR functions first
        tir_mod = tvm_tirx.IRModule(
            {
                gv: func
                for gv, func in self.ir_mod.functions_items()
                if isinstance(func, tirx.Function)
            }
        )
        if tir_mod:
            try:
                tir_exec_mod = tvm.compile(tir_mod, target=self.target)
                for func_name in self.tir_func_names:
                    self.compiled_tir_funcs[func_name] = tir_exec_mod[func_name]
            # pylint: disable=broad-exception-caught
            except Exception as error:
                print(f"Warning: Failed to compile one or more TIR functions: {error}")

        if self.relax_func_names:
            try:
                exec_mod = tvm.compile(self.ir_mod, target=self.target)
                self.relax_vm = relax.VirtualMachine(exec_mod, self.device)
            # pylint: disable=broad-exception-caught
            except Exception as error:
                print(f"Warning: Failed to compile Relax VM: {error}")
                self.relax_vm = None

    def _wrap_tir_functions(self):
        """Wrap TIR functions to make them accessible as instance attributes."""
        for func_name, func in self.compiled_tir_funcs.items():
            setattr(self, func_name, func)

    def _wrap_relax_functions(self):
        """Wrap Relax functions to be callable from Python with auto conversion."""
        for func_name in self.relax_func_names:

            def _create_relax_wrapper(name):
                def wrapper(*args, **kwargs):
                    """Wrapper for Relax function with automatic tensor conversion."""
                    if hasattr(self.ir_mod, "__pyfuncs__") and name in self.ir_mod.__pyfuncs__:
                        return self.ir_mod.__pyfuncs__[name](*args, **kwargs)

                    if self.relax_vm is not None:
                        converted_args = self._convert_pytorch_to_tvm(list(args))
                        converted_kwargs = {
                            k: self._convert_pytorch_to_tvm(v) for k, v in kwargs.items()
                        }
                        result = self.relax_vm[name](*converted_args, **converted_kwargs)
                        return self._convert_tvm_to_pytorch(result)

                    raise RuntimeError(
                        f"Neither converted Python function nor Relax VM available for {name}"
                    )

                wrapper.__name__ = name
                wrapper.__doc__ = f"Wrapped Relax function: {name}"
                return wrapper

            setattr(self, func_name, _create_relax_wrapper(func_name))

    def _register_python_functions(self):
        """Register Python functions with the VM runtime for call_py_func support."""
        if not hasattr(self.ir_mod, "__pyfuncs__") or not self.ir_mod.__pyfuncs__:
            return

        try:
            register_py_func = tvm_ffi.get_global_func("vm.builtin.register_py_func")
        except ValueError:
            return

        for func_name, py_func in self.ir_mod.__pyfuncs__.items():

            def create_py_func_wrapper(name, original_func):
                # The registry owns the wrapper, so capture the module weakly. A strong capture
                # would make every registering module immortal, keeping its entries registered
                # for the lifetime of the process.
                module_ref = weakref.ref(self)

                def wrapper(*args, **kwargs):
                    module = module_ref()
                    if module is None:
                        raise RuntimeError(
                            f"Python function '{name}' belongs to a BasePyModule that has been "
                            "destroyed; keep the module alive while calling into it."
                        )

                    converted_args = [module._convert_tvm_to_pytorch(arg) for arg in args]
                    converted_kwargs = {
                        k: module._convert_tvm_to_pytorch(v) for k, v in kwargs.items()
                    }

                    result = original_func(module, *converted_args, **converted_kwargs)

                    return module._convert_pytorch_to_tvm(result)

                wrapper.__name__ = name
                return wrapper

            wrapped_func = create_py_func_wrapper(func_name, py_func)
            register_py_func(func_name, wrapped_func)
            _PY_FUNC_OWNERS[func_name] = id(self)
            if func_name not in self._registered_py_funcs:
                self._registered_py_funcs = (*self._registered_py_funcs, func_name)

    def call_tir(self, tir_func, args, out_ty):
        """Call a TIR function with PyTorch tensors."""
        # Try to get function name from different sources
        if isinstance(tir_func, str):
            func_name = tir_func
        elif hasattr(tir_func, "name"):
            func_name = tir_func.name
        elif hasattr(tir_func, "__name__"):
            func_name = tir_func.__name__
        else:
            # Try to find by function object reference
            for name, func in self.compiled_tir_funcs.items():
                if func == tir_func:
                    func_name = name
                    break
            else:
                func_name = None

        if not func_name or func_name not in self.compiled_tir_funcs:
            available_funcs = list(self.compiled_tir_funcs.keys())
            raise ValueError(
                f"Could not resolve or find compiled TIR function: {tir_func}. "
                f"Available functions: {available_funcs}"
            )
        func = self.compiled_tir_funcs[func_name]

        out = self._create_output_tensors(out_ty, args)
        tvm_args = self._convert_pytorch_to_tvm(args)
        tvm_out = self._convert_pytorch_to_tvm(out)

        func(*tvm_args, *tvm_out)

        result = self._convert_tvm_to_pytorch(tvm_out)
        return result[0] if len(result) == 1 else result

    def call_dps_packed(self, func_name: str, args, out_ty):
        """Call a packed function with PyTorch tensors, converting TVM Tensors via DLPack."""
        if hasattr(self, func_name) and callable(getattr(self, func_name)):
            return getattr(self, func_name)(*args)

        if func_name not in self.extern_funcs:
            try:
                self.extern_funcs[func_name] = tvm_ffi.get_global_func(func_name)
            except ValueError as error:
                raise ValueError(
                    f"Function '{func_name}' not found as a global function. "
                    f"Please implement it as a method or register it."
                ) from error
        func = self.extern_funcs[func_name]

        out = self._create_output_tensors(out_ty, args)
        tvm_args = self._convert_pytorch_to_tvm(args)
        tvm_out = self._convert_pytorch_to_tvm(out)
        func(*tvm_args, *tvm_out)
        return out[0] if len(out) == 1 else out

    def call_py_func(self, func_name: str, args):
        """Call a Python function stored in the module's pyfuncs."""
        if func_name not in self.pyfuncs:
            raise ValueError(f"Python function '{func_name}' not found in module pyfuncs")
        py_func = self.pyfuncs[func_name]
        return py_func(self, *args)

    def _create_output_tensors(self, out_ty, in_args=None):
        # pylint: disable=import-outside-toplevel
        import torch

        ty_list = out_ty if isinstance(out_ty, list) else [out_ty]
        out_tensors = []
        for ty in ty_list:
            if isinstance(ty, tuple | list) and all(isinstance(x, int | np.integer) for x in ty):
                out_tensors.append(torch.zeros(list(map(int, ty)), dtype=torch.float32))
                continue

            if hasattr(ty, "shape") and hasattr(ty, "dtype"):
                concrete_shape = self._infer_concrete_shape_from_args(ty.shape, in_args)
                torch_dtype = self._convert_tvm_dtype_to_torch(ty.dtype)
                out_tensors.append(torch.zeros(concrete_shape, dtype=torch_dtype))
                continue

            out_tensors.append(torch.zeros((1,), dtype=torch.float32))
        return out_tensors

    def _infer_concrete_shape_from_args(self, shape, in_args):
        concrete = []
        symbolic_positions = []
        for idx, dim in enumerate(shape):
            if isinstance(dim, int | np.integer):
                concrete.append(int(dim))
            elif isinstance(dim, tvm_tirx.ir.IntImm):
                concrete.append(int(dim.value))
            else:
                concrete.append(None)
                symbolic_positions.append(idx)

        if not symbolic_positions:
            return concrete

        candidates = []
        if in_args is not None:
            if not isinstance(in_args, list | tuple):
                in_args = [in_args]
            for obj in in_args:
                if hasattr(obj, "shape") and isinstance(obj.shape, tuple | list):
                    try:
                        candidates.append(tuple(int(x) for x in obj.shape))
                        continue
                    except (ValueError, TypeError):
                        # Skip objects with invalid shapes
                        pass

        target_ndim = len(shape)
        for cand in candidates:
            if len(cand) == target_ndim:
                for pos in symbolic_positions:
                    concrete[pos] = cand[pos]
                if all(x is not None for x in concrete):
                    return concrete

        raise ValueError(
            "Cannot infer concrete output shape from symbolic shape and inputs. "
            "Please provide a concrete `out_ty` (e.g., a tuple/list of ints) "
            "or ensure input tensors carry shapes that determine output extents."
        )

    def _convert_tvm_dtype_to_torch(self, tvm_dtype: str) -> "torch.dtype":
        """Convert TVM dtype string to PyTorch dtype."""
        # pylint: disable=import-outside-toplevel
        import torch

        dtype_mapping = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        return dtype_mapping.get(str(tvm_dtype), torch.float32)

    def _convert_pytorch_to_tvm(
        self, tensors: Any | list[Any] | tuple[Any, ...]
    ) -> Tensor | list[Tensor]:
        """Convert PyTorch tensors to TVM Tensors using DLPack."""
        # pylint: disable=import-outside-toplevel
        import torch

        if isinstance(tensors, list | tuple):
            return [self._convert_single_pytorch_to_tvm(t) for t in tensors]
        return self._convert_single_pytorch_to_tvm(tensors)

    def _convert_single_pytorch_to_tvm(self, tensor: Any) -> Tensor:
        """Convert a single PyTorch tensor to TVM Tensor with faster DLPack converter."""
        # pylint: disable=import-outside-toplevel
        import torch

        if isinstance(tensor, Tensor):
            return tensor
        if isinstance(tensor, torch.Tensor):
            # 1. Try faster C++ DLPack converter
            if _FASTER_DLPACK_EXTENSION is not None:
                try:
                    dlpack = torch.to_dlpack(tensor)
                    return tvm_ffi.from_dlpack(dlpack)
                except (AttributeError, ValueError):
                    pass  # Fall through to the next method

            # 2. Try modern `torch.to_dlpack` (preferred for PyTorch >= 1.7)
            try:
                dlpack = torch.to_dlpack(tensor)
                return tvm_ffi.from_dlpack(dlpack)
            except (AttributeError, ValueError):
                pass  # Fall through to the next method

            # 3. Try legacy `torch.utils.dlpack.to_dlpack`
            if to_dlpack_legacy:
                try:
                    dlpack = to_dlpack_legacy(tensor)
                    return tvm_ffi.from_dlpack(dlpack)
                except (AttributeError, ValueError) as error_legacy:
                    print(
                        f"Warning: Legacy DLPack conversion failed ({error_legacy}), "
                        f"using numpy fallback."
                    )

            # 4. If all DLPack methods fail, use numpy fallback
            numpy_array = tensor.detach().cpu().numpy()
            return tvm_tirx.runtime.tensor(numpy_array, device=self.device)

        # For other types (like scalars, lists), convert to numpy first
        try:
            numpy_array = np.array(tensor, dtype=np.float32)
            return tvm_tirx.runtime.tensor(numpy_array, device=self.device)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Unsupported type for conversion to TVM Tensor: {type(tensor)}"
            ) from error

    def _convert_tvm_to_pytorch(
        self, tvm_tensors: Any | list[Any]
    ) -> Union["torch.Tensor", list["torch.Tensor"]]:
        """Convert TVM Tensors to PyTorch tensors using DLPack."""
        if isinstance(tvm_tensors, list | tuple):
            return [self._convert_single_tvm_to_pytorch(tensor) for tensor in tvm_tensors]
        return self._convert_single_tvm_to_pytorch(tvm_tensors)

    def _convert_single_tvm_to_pytorch(self, tvm_tensor: Any) -> "torch.Tensor":
        """Convert a single TVM Tensor to PyTorch tensor using faster DLPack converter."""
        # pylint: disable=import-outside-toplevel
        import torch

        if isinstance(tvm_tensor, torch.Tensor):
            return tvm_tensor
        if not isinstance(tvm_tensor, Tensor):
            return torch.tensor(tvm_tensor)

        # 1. Try faster C++ DLPack converter
        if _FASTER_DLPACK_EXTENSION is not None:
            try:
                return torch.from_dlpack(tvm_tensor)
            except (AttributeError, ValueError):
                pass  # Fall through to the next method

        # 2. Try standard DLPack conversion
        try:
            return torch.from_dlpack(tvm_tensor)
        # pylint: disable=broad-exception-caught
        except Exception as error:
            print(f"Warning: DLPack conversion from TVM failed ({error}), using numpy fallback")
            numpy_array = tvm_tensor.numpy()
            return torch.from_numpy(numpy_array)

    def get_function(self, name: str) -> Function | None:
        """Get a compiled function by name."""
        if name in self.compiled_tir_funcs:
            return self.compiled_tir_funcs[name]
        if name in self.extern_funcs:
            return self.extern_funcs[name]
        if self.relax_vm and name in self.relax_func_names:
            try:
                if hasattr(self, name):
                    return getattr(self, name)
                return self.relax_vm[name]
            except AttributeError as error:
                print(f"Warning: Failed to get Relax function '{name}': {error}")
        return None

    def list_functions(self) -> dict[str, list[str]]:
        """List all available functions."""
        return {
            "tirx": self.tir_func_names,
            "relax": self.relax_func_names,
            "extern": list(self.extern_funcs.keys()),
        }

    def add_python_function(self, name: str, func: callable):
        """Add a Python function to the module."""
        self.pyfuncs[name] = func

        # Create a wrapper that handles both instance methods and static functions
        # pylint: disable=import-outside-toplevel
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if params and params[0] == "self":
                return func(self, *args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Set the wrapper as an instance attribute
        setattr(self, name, wrapper)

    def script(
        self,
        *,
        name: str | None = None,
        show_meta: bool = False,
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int = -1,
        show_all_ty: bool = True,
        extra_config: dict | None = None,
    ) -> str:
        """Print TVM IR into TVMScript text format with Python function support.

        This method extends the standard IRModule script() method to handle
        Python functions stored in the IRModule's ``__pyfuncs__`` attribute.
        """
        merged_extra = {"relax.show_all_ty": show_all_ty, **(extra_config or {})}
        # First get the standard IRModule script
        base_script = self.ir_mod.script(
            name=name,
            show_meta=show_meta,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            verbose_expr=verbose_expr,
            indent_spaces=indent_spaces,
            print_line_numbers=print_line_numbers,
            num_context_lines=num_context_lines,
            extra_config=merged_extra,
        )

        # If there are no Python functions, return the base script
        if not hasattr(self.ir_mod, "__pyfuncs__") or not self.ir_mod.__pyfuncs__:
            return base_script

        # Insert Python functions into the script
        return self._insert_python_functions(base_script, indent_spaces)

    def _insert_python_functions(self, base_script: str, indent_spaces: int) -> str:
        """Insert Python functions into the TVMScript output."""
        lines = base_script.split("\n")
        result_lines = []

        # Find the class definition line and insert Python functions after it
        class_found = False
        class_indent = 0

        for line in lines:
            result_lines.append(line)

            # Look for class definition
            if not class_found and line.strip().startswith("class "):
                class_found = True
                class_indent = len(line) - len(line.lstrip())

                # Insert Python functions after the class definition
                if hasattr(self.ir_mod, "__pyfuncs__") and self.ir_mod.__pyfuncs__:
                    for func_name, func in self.ir_mod.__pyfuncs__.items():
                        # Get the function source code
                        func_source = self._get_function_source(func)
                        if func_source:
                            # Format the function with proper indentation
                            formatted_func = self._format_python_function(
                                func_name, func_source, class_indent + indent_spaces
                            )
                            result_lines.append(formatted_func)
                            result_lines.append("")  # Add empty line for separation

        return "\n".join(result_lines)

    def _get_function_source(self, func: callable) -> str | None:
        """Get the source code of a Python function."""
        try:
            source = inspect.getsource(func)
            return source
        except (OSError, TypeError):
            # If we can't get the source, return None
            return None

    def _format_python_function(self, _func_name: str, func_source: str, indent: int) -> str:
        """Format a Python function with proper indentation for TVMScript."""
        lines = func_source.split("\n")
        formatted_lines = []

        for line in lines:
            # Skip the function definition line if it's already properly indented
            if line.strip().startswith("def ") or line.strip().startswith("@"):
                # Keep decorators and function definition as is
                formatted_lines.append(" " * indent + line.strip())
            else:
                # Add proper indentation for the function body
                formatted_lines.append(" " * indent + line.strip())

        return "\n".join(formatted_lines)

    def show(self, style: str | None = None, **kwargs) -> None:
        """A sugar for print highlighted TVM script with Python function support.

        This method extends the standard IRModule show() method to handle
        Python functions stored in the IRModule's pyfuncs attribute.
        """
        from tvm_tirx.script.printer.highlight import (
            cprint,  # pylint: disable=import-outside-toplevel
        )

        script_content = self.script(**kwargs)
        cprint(script_content, style=style)
