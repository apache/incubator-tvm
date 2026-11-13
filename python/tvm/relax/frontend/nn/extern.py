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
# ruff: noqa: E722
"""External modules to be linked into the exported IRModule."""

import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import tvm_ffi

import tvm
import tvm_tirx
from tvm_tirx import libinfo
from tvm_tirx.codegen import load_static_library
from tvm_tirx.support import cc as _cc

from ...op import call_dps_packed
from . import core
from .core import wrap_nested


class ExternModule:
    """The abstract base class for external modules. External modules are designed to help
    incorporate user-provided handcrafted kernels into the exported TVM IRModule.
    """

    _symbols: dict[str, Callable]

    def __init__(self, symbols: dict[str, Callable]) -> None:
        self._symbols = symbols

    def __getitem__(self, func_name: str) -> Callable:
        _inference_function = self._symbols[func_name]

        def _call(*input_args):
            def _convert(arg, name: str):
                from tvm import relax as rx  # pylint: disable=import-outside-toplevel

                if isinstance(arg, core.Tensor):
                    return arg._expr  # pylint: disable=protected-access
                if isinstance(arg, int):
                    return rx.prim_value(tvm_tirx.ir.IntImm("int64", arg))
                if isinstance(arg, float):
                    return rx.prim_value(tvm_tirx.ir.FloatImm("float64", arg))
                if isinstance(arg, str):
                    return tvm_tirx.ir.StringImm(arg)
                if tvm_tirx.ir.is_prim_expr(arg):
                    return rx.prim_value(arg)
                if isinstance(arg, tuple | list):
                    return rx.Tuple([_convert(e, f"{name}_{i}") for i, e in enumerate(arg)])
                raise TypeError(f"Unsupported input type: {type(arg)}")

            rx_inputs = _convert(input_args, "input")
            rx_outputs_ty = _convert(_inference_function(*input_args), "dummy").ty
            return wrap_nested(
                call_dps_packed(
                    tvm.relax.ExternFunc(func_name), rx_inputs, ty_args=[rx_outputs_ty]
                ),
                func_name,
            )

        return _call

    def _load(self, path: Path) -> tvm_ffi.Module:
        return load_static_library(str(path), func_names=list(self._symbols.keys()))

    def load(self) -> tvm_ffi.Module:
        """Loads the external module into a TVM runtime module."""
        raise NotImplementedError


class ObjectModule(ExternModule):  # pylint: disable=too-few-public-methods
    """A subclass of `nn.ExternModule`, which allows
    users to provide an object `.o` file to be linked into compiled
    artifact;
    """

    def __init__(
        self,
        symbols: dict[str, Callable],
        filepath: Path,
    ) -> None:
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        if not filepath.is_file():
            raise ValueError(f"Not a file: {filepath!s}")
        self.filepath = filepath
        super().__init__(symbols)

    def load(self) -> tvm_ffi.Module:
        return self._load(self.filepath)


class SourceModule(ExternModule):  # pylint: disable=too-few-public-methods
    """A subclass of `nn.ExternModule`. It compiles C++/CUDA source code and link them into the
    eventual IRModule.

    **Shape/dtype inference.** The `nn.ExternModule` system requires users to provide additional
    information to work, namely, `symbols`. It is a dictionary that maps each symbol in the
    external object file to its shape/dtype inference function. Consider a case where function
    `my_func` accepts two tensors, `a` of shape `(x, y, 1)`, and `b` of shape `(y, z, 5)`, and
    produces a tensor `c` of shape `(x, y, z, 9)`, the shape/dtype inference function should look
    like:

    .. code-block:: python

        def shape_dtype_inference(a, b):
            x, y, _ = a.shape
            _, z, _ = b.shape
            return nn.Tensor.placeholder((x, y, z, 9), dtype="float32")


    and the `symbols` dictionary should be provided as:

    .. code-block:: python

        symbols={
            "my_func": shape_dtype_inference,
        }


    **Calling convention.** All external modules now follows "destination-passing-style" (DPS)
    calling convention, which means the returned tensors are pre-allocated by the system already
    and passed in as an argument of the external function.

    Reuse the example above, the implementation of `my_func` should include three parameters in
    its signature, where tensors are represented using DLTensor from DLPack, the de facto standard
    of in-memory representation of tensors. More details:
    https://github.com/dmlc/dlpack/blob/v0.8/include/dlpack/dlpack.h#L163-L206.

    To expose the symbol, `TVM_FFI_DLL_EXPORT_TYPED_FUNC(symbol, function)` is guaranteed available:

    .. code-block:: C++

        // those headers are guaranteed to be available
        #include <dlpack/dlpack.h>
        #include <tvm/ffi/dtype.h>
        #include <tvm/ffi/function.h>

        namespace {
        // anonymous namespace hides the symbol `_my_func_impl` from other translation units
        int _my_func_impl(DLTensor* a, DLTensor* b, DLTensor* c) {
            // `a` and `b` are inputs, and `c` is the output
        }
        }
        // expose symbol `my_func` instead of `_my_func_impl`
        TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_func, _my_func_impl);

    **A compiler pass `AttachExternModules`.** It is introduced to attach a list of
    `nn.ExternModule`s into an IRModule at any stage of the compilation pipeline,
    and attach the compiled external modules as `tvm_ffi.Module`s into IRModule's `external_mods`
    attribute. It is required by linking in `tvm.compile`, but with the existence of this pass,
    source compilation can be deferred to arbitrary stage of TVM compilation.

    **Caveats.** It is required to call `nn.add_extern` to register external modules exactly once
    during `export_tvm`. Each symbol should be registered exactly once to avoid potential conflicts,
    and otherwise an error will be raised.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        symbols: dict[str, Callable],
        source_code: str | Path,
        source_format: str,  # "cpp", "cu"
        compile_options: list[str] | None = None,
        compiler: str | None = None,
        output_format: str = "obj",  # "obj", "wasm"
    ):
        """Constructs a `nn.SourceModule` from source code.

        Parameters
        ----------
        symbols : Dict[str, Callable]
            The dictionary that maps each symbol in the external object file to its shape/dtype
            inference function.

        source_code : Union[str, Path]
            Source code or path to the source code to be compiled.

        source_format : str
            The source code format. It can be either "cpp" or "cu".

        compile_options : Optional[List[str]]
            The compile options. If not provided, the default compile options will be used.

        compiler : Optional[str]
            The compiler. If not provided, the default compiler will be used. On Windows,
            compilation requires `clang` by default.

        output_format : str
            The output format. It can be either "obj" or "wasm". "obj" is the default format,
            which is a shared object file. "wasm" is the WebAssembly format, which is a binary
            file.
        """

        def _detect_input_suffix(source_format: str) -> str:
            if source_format == "cpp":
                return ".cpp"
            if source_format == "cu":
                return ".cu"
            raise ValueError(f"Invalid source format: {source_format}")

        def _detect_output_suffix(output_format: str) -> str:
            if output_format == "obj":
                if _cc._is_linux_like():  # pylint: disable=protected-access
                    return ".o"
                if _cc._is_windows_like():  # pylint: disable=protected-access
                    return ".obj"
                raise ValueError(f"Unsupported platform: {sys.platform}")
            if output_format == "wasm":
                return ".wasm"
            raise ValueError(f"Invalid output format: {output_format}")

        def _detect_source_code(source_code) -> str:
            if isinstance(source_code, Path):
                path = source_code
                if not path.is_file():
                    raise ValueError(f"Not a file: {path!s}")
            else:
                try:
                    path = Path(source_code)
                except:  # pylint: disable=bare-except
                    return source_code
                try:
                    if not path.is_file():
                        return source_code
                except:  # pylint: disable=bare-except
                    return source_code
            with path.open("r", encoding="utf-8") as file:
                return file.read()

        self.source_code = _detect_source_code(source_code)
        if compile_options is None:
            self.compile_options = SourceModule.get_compile_options(source_format=source_format)
        else:
            self.compile_options = list(compile_options)
        self.compiler = compiler
        self.source_suffix = _detect_input_suffix(source_format)
        self.output_suffix = _detect_output_suffix(output_format)
        super().__init__(symbols)

    @staticmethod
    def tvm_home() -> Path:
        """Find TVM's home directory. If `TVM_HOME` environment variable is set, use it.
        Otherwise, use the directory where the `tvm` Python package is installed.
        As a sanity check, it is required to have `include` and `3rdparty` as direct subdirectories.

        Returns
        -------
        tvm_home : pathlib.Path
            The TVM home directory, and it is guaranteed to have `include` and `3rdparty` as
            direct subdirectories.
        """
        if os.environ.get("TVM_HOME", None):
            tvm_path = Path(os.environ["TVM_HOME"])
            assert tvm_path.exists(), (
                f"Using environment variable `TVM_HOME`, but directory not found: {tvm_path!s}"
            )
            assert tvm_path.is_dir(), (
                f"Using environment variable `TVM_HOME`, but it is not a directory: {tvm_path!s}"
            )
        else:
            import tvm

            tvm_path = Path(tvm.__file__).parent
            assert tvm_path.is_dir()
        tvm_path = tvm_path.resolve()
        while True:
            exists_include = (tvm_path / "include").is_dir()
            exists_3rdparty = (tvm_path / "3rdparty").is_dir()
            if exists_include and exists_3rdparty:
                return tvm_path.resolve()
            parent = tvm_path.parent
            if parent == tvm_path:
                raise ValueError(
                    "Cannot detect TVM directory. "
                    "Please explicitly specify it by setting `TVM_HOME` environment variable, "
                    "and make sure it contains `include` and `3rdparty` as direct sub-directories."
                )
            tvm_path = parent
        return tvm_path.resolve()

    @staticmethod
    def get_includes(tvm_pkg: list[str] | None = None) -> list[Path]:
        """Returns the default include paths according to `tvm_home()`.
        By default, it includes TVM, DLPack. With `tvm_pkg` provided, it also
        includes the specified package under `tvm_home/3rdparty`.

        Parameters
        ----------
        tvm_pkg : Optional[List[str]]
            The list of packages to be included under `tvm_home/3rdparty`. Each element should be
            a relative path to `tvm_home/3rdparty`.

        Returns
        -------
        includes : List[pathlib.Path]
            The list of include paths.
        """
        results = [
            Path(tvm.libinfo.find_include_path()),
            Path(libinfo.find_include_path()),
            Path(tvm_ffi.libinfo.find_include_path()),
            Path(tvm_ffi.libinfo.find_dlpack_include_path()),
        ]
        if tvm_pkg:
            tvm_home = SourceModule.tvm_home()
            for relative in tvm_pkg:
                results.append(tvm_home / "3rdparty" / relative)
        results = list(dict.fromkeys(results))
        for path in results:
            assert path.exists(), f"Not found: {path!s}"
            assert path.is_dir(), f"Not a directory: {path!s}"
        return results

    @staticmethod
    def get_compile_options(
        source_format: str,
        tvm_pkg: list[str] | None = None,
    ) -> list[str]:
        """Returns the default compile options depending on `source_format`, including the default
        inlcude paths w.r.t. `tvm_home()`, and by default,
        it uses "-O3" and "-std=c++17".

        Parameters
        ----------
        source_format : str
            The source code format. It can be either "cpp" or "cu".

        tvm_pkg : Optional[List[str]]
            The list of packages to be included under `tvm_home/3rdparty`. Each element should be
            a relative path to `tvm_home/3rdparty`.

        Returns
        -------
        compile_options : List[str]
            The list of compilation flags.
        """
        include_flags = []
        for include_path in SourceModule.get_includes(tvm_pkg=tvm_pkg):
            include_flags += ["-I", str(include_path)]
        if source_format == "cpp":
            host_flags = [
                "-c",  # generate object file
                "-O3",
                "-std=c++17",
            ]
        elif source_format == "cu":
            host_flags = [
                "-c",  # generate object file
                "-O3",
                "-std=c++17",
                # Enable `-fPIC` for the host compiler
                "-Xcompiler=-fPIC",
            ]
        else:
            raise ValueError(f"Invalid source format: {source_format}")
        return include_flags + host_flags

    def compile(self, output_path: Path) -> None:
        """Compiles the source code in a provided directory and returns the compiled artifact."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_filename = f"main{self.source_suffix}"
            object_filename = f"main{self.output_suffix}"
            source_path = temp_dir / source_filename
            object_path = temp_dir / object_filename
            with source_path.open("w", encoding="utf-8") as file:
                file.write(self.source_code)
            _cc.create_shared(
                output=object_filename,
                objects=[source_filename],
                options=self.compile_options,
                cc=self.compiler,
                cwd=temp_dir,
                ccache_env=(
                    {
                        "CCACHE_COMPILERCHECK": "content",
                        "CCACHE_NOHASHDIR": "1",
                    }
                    if shutil.which("ccache")
                    else None
                ),
            )
            shutil.move(str(object_path), str(output_path))

    def load(self) -> tvm_ffi.Module:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_path = Path(temp_dir_str) / f"main{self.output_suffix}"
            self.compile(output_path)
            return self._load(output_path)
