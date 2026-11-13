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
# pylint: disable=invalid-name, redefined-builtin, no-else-return, consider-using-dict-items
# ruff: noqa: RUF005
"""The Relax virtual machine."""

from collections.abc import Callable
from enum import IntEnum
from numbers import Integral, Number
from typing import TYPE_CHECKING, Any

import numpy as np  # type: ignore
import tvm_ffi
from tvm_ffi import Function, register_global_func

import tvm_tirx
from tvm_tirx.runtime import Device

from ..benchmark import BenchmarkResult, time_evaluator

if TYPE_CHECKING:
    from tvm_tirx.codegen import Executable


class VMInstrumentReturnKind(IntEnum):
    NO_OP = 0
    # skip the following call, only valid in before
    SKIP_RUN = 1


class VirtualMachine:
    """Relax VM runtime."""

    def __init__(
        self,
        rt_mod: "tvm_ffi.Module | Executable",
        device: Device | list[Device],
    ) -> None:
        """
        Construct a VirtualMachine wrapper object.

        Parameters
        ----------
        rt_mod: Union[tvm_ffi.Module, tvm_tirx.codegen.Executable]
            Runtime module exported by the result of build.

        device : Union[Device, List[Device]]
            The device to deploy the module.
        """
        if not isinstance(rt_mod, tvm_ffi.Module):
            from tvm_tirx.codegen import Executable  # pylint: disable=import-outside-toplevel

            if isinstance(rt_mod, Executable):
                rt_mod = rt_mod.jit()
            else:
                raise ValueError("Expect the rt_mod to be an tvm_ffi.Module")

        self.module = rt_mod["vm_load_executable"]()
        self._invoke_closure = self.module["invoke_closure"]
        self._save_function = self.module["save_function"]
        self._set_input = self.module["set_input"]
        self._invoke_stateful = self.module["invoke_stateful"]
        self._get_output = self.module["get_output"]
        self._get_output_arity = self.module["get_output_arity"]
        self._get_function_arity = self.module["get_function_arity"]
        self._get_function_param_name = self.module["get_function_param_name"]
        self._set_instrument = self.module["set_instrument"]
        self._setup_device(device)

    def _setup_device(self, dev: Device | list[Device]) -> None:
        """Initialize devices and owning tensor pools."""
        devs = dev
        if not isinstance(dev, list | tuple):
            if not isinstance(dev, tvm_tirx.runtime.Device):
                raise TypeError("dev is expected to be Device or List[Device]")
            devs = [dev]

        # CPU is required for executing shape functions
        if devs[-1].dlpack_device_type() != tvm_tirx.cpu().dlpack_device_type():
            devs.append(tvm_tirx.cpu())

        init_args = []
        for device in devs:
            init_args.append(device.dlpack_device_type())
            init_args.append(device.index)
        self.module["vm_initialization"](*init_args)

    def __getitem__(self, key: str) -> Function:
        return self.module[key]

    def invoke_closure(self, closure: tvm_ffi.Object, *args: Any) -> tvm_ffi.Object:
        """Invoke a closure.

        Parameters
        ----------
        closure : tvm_ffi.Object
            The VMClosure Object.

        args : list[tvm_tirx.runtime.Tensor] or list[np.ndarray]
            The arguments to the closure.

        Returns
        -------
        result : tvm_ffi.Object
            The output.
        """
        return self._invoke_closure(closure, *args)

    def save_function(
        self,
        func_name: str,
        saved_name: str,
        *args: list[Any],
        include_return: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Convenience function. Takes a function from the module and saves
        a `Function` that, when called, will invoke the function with the given arguments.
        The `Function` can be accessed from the module using `saved_name`.
        This is included to facilitate timing trials:
        Invoking the returned `Function` will have less overhead from dictionary lookups
        than normally running through the VM.

        If the saved name is taken, it can be overridden, though it cannot override
        the name of a function defined in the Relax source.

        This is really creating a closure, but the function has a different name
        to avoid confusion with `invoke_closure` (they are not meant to be used together).

        Parameters
        ----------
        func_name : str
            The function that should be packaged up.

        saved_name : str
            The name that the resulting closure should be saved under.

        include_return : bool
            Whether the saved Function should return its output.
            If timing over RPC, it may not be desirable to send output
            between machines.

        args : List[Any]
            The arguments to package up with the function.

        kwargs : Dict[str, Any]
            Any named arguments to package up with the function
        """
        cargs: list[Any] = []
        if kwargs:
            args = self._convert_func_named_args(func_name, args, **kwargs)
        for arg in args:
            self._convert(arg, cargs)
        self._save_function(func_name, saved_name, int(include_return), *cargs)

    def _convert(self, arg: Any, cargs: list) -> None:
        """helper function to convert arguments to vm function."""

        def _gettype(arg):
            if isinstance(arg, np.float16):
                return "float16"
            elif isinstance(arg, Integral | bool):
                return "int32"
            else:
                return "float32"

        if isinstance(arg, tvm_ffi.Object):
            cargs.append(arg)
        elif isinstance(arg, np.ndarray):
            nd_arr = tvm_tirx.runtime.tensor(arg, device=tvm_tirx.cpu(0))
            cargs.append(nd_arr)
        elif isinstance(arg, tvm_tirx.runtime.Tensor):
            cargs.append(arg)
        elif isinstance(arg, tuple | list):
            field_args: list[Any] = []
            for field in arg:
                self._convert(field, field_args)
            cargs.append(tuple(field_args))
        elif isinstance(arg, Number | bool):
            dtype = _gettype(arg)
            value = tvm_tirx.runtime.tensor(np.array(arg, dtype=dtype), device=tvm_tirx.cpu(0))
            cargs.append(value)
        elif isinstance(arg, str):
            cargs.append(arg)
        else:
            raise TypeError(f"Unsupported type: {type(arg)}")

    def _convert_func_named_args(self, func_name: str, args: Any, **kwargs: Any) -> Any:
        """
        Takes named function parameters and returns a list of those needed,
        in the order they should appear
        """
        # kwargs can be a super set of the required function parameters.
        # We only find the ones that are needed.
        func_arity = self._get_function_arity(func_name)
        func_params = [self._get_function_param_name(func_name, i) for i in range(func_arity)]
        new_args = [None] * len(func_params)
        cnt = 0
        for k in kwargs:
            if k in func_params:
                idx = func_params.index(k)
                new_args[idx] = kwargs[k]
                cnt += 1
            else:
                print(f'Warning: Keyword argument "{k}" is unused in {func_name}')
        assert len(args) + cnt == len(func_params)
        idx = 0
        for i, arg in enumerate(new_args):
            if arg is None:
                new_args[i] = args[idx]
                idx += 1
        return new_args

    def set_input(self, func_name: str, *args: Any, **kwargs: Any) -> None:
        """Set the inputs to a function.
        This interface works when using VM over RPC by internally converting Tensor in
        the arguments to DLTensor, which is supported in RPC where remote could only
        have a minimal C runtime.

        Note: If `set_input` is used, the function *must* be called using `invoke_stateful`
        and the results must be obtained using `get_outputs`.

        Parameters
        ----------
        func_name : str
            The name of the function.
        args: List[tvm_tirx.runtime.Tensor] or List[np.ndarray]
            The arguments to the function.
        kwargs: dict of str to tvm_tirx.runtime.Tensor or np.ndarray
            Named arguments to the function.
        """
        cargs: list[Any] = []

        if kwargs:
            args = self._convert_func_named_args(func_name, args, **kwargs)

        for arg in args:
            self._convert(arg, cargs)

        self._set_input(func_name, *cargs)

    def invoke_stateful(self, func_name: str) -> None:
        """
        Call the named function from the VM module using the arguments set using `set_input`.
        It is an error to call `invoke_stateful` without using `set_input` first
        (even if it's to set 0 inputs); conversely, if `set_input` has been called,
        it is an error to call the function without using `invoke_stateful`.

        The results of the call can be obtained by calling `get_outputs`.

        Parameters
        ----------
        func_name: str
            The name of the function to call.
        """
        self._invoke_stateful(func_name)

    def get_outputs(self, func_name: str) -> tvm_ffi.Object | tuple[Any]:
        """
        Get the value output by the function by the given name
        after a call of `invoke_stateful`.

        It is an error to call this function without first calling `invoke_stateful`.

        Parameters
        ----------
        func_name: str
            The name of the function whose output should be fetched.

        Returns
        -------
        ret: Union[tvm_ffi.Object, Tuple[Any]]
            The result of the earlier call to the function via `invoke_stateful`.
            If the result is a tuple, it returns a list of the fields.
            The fields are potentially also tuples, so these can be arbitrily nested.
        """

        # to deal with potentially nested tuples, we need to query for arity recursively
        def get_output_rec(func_name, *idx):
            arity = self._get_output_arity(func_name, *idx)
            if arity == -1:
                return self._get_output(func_name, *idx)
            # otherwise we need to specify more indices
            idx_list = list(idx)
            return tuple(get_output_rec(func_name, *(idx_list + [i])) for i in range(arity))

        return get_output_rec(func_name)

    def set_instrument(self, instrument: Function) -> None:
        """Set an instrumentation function.

        If instrument is present, the function will be called
        before/after each Call instruction. The function have
        the following signature:

        .. code:: python

            def instrument(
                func: Union[VMClosure, Function],
                func_symbol: str,
                before_run: bool,
                ret_value: any,
                *args) -> bool:
                pass

        The instrument takes the following parameters:
        - func: function object to be called.
        - func_symbol: the symbol name of the function.
        - before_run: whether it is before or after call.
        - ret_value: the return value of the call, only valid after run.
        - args: the arguments being passed to call.

        The instrument function can choose an integer,
        which corresponds to action direction for the
        following run. See VMInstrumentReturnKind for
        more details.

        Parameters
        ----------
        instrument: tvm_ffi.Function
            A instrumentation function that get invoked every VM call instr.

        See Also
        --------
        VMInstrumentReturnKind: the possible return values in VM.
        """
        self._set_instrument(instrument)

    def time_evaluator(
        self,
        func_name: str,
        dev: Device,
        number: int = 10,
        repeat: int = 1,
        min_repeat_ms: int = 0,
        cooldown_interval_ms: int = 0,
        repeats_to_cooldown: int = 1,
        f_preproc: str = "",
    ) -> Callable[..., BenchmarkResult]:
        """
        Returns an evaluator that times a function in the module.
        This follows the same convention as tvm.runtime.time_evaluator.
        This can be used in combination with save_function() so that the
        timings avoid extra dictionary lookups.

        Parameters
        ----------
        func_name: str
            The name of the function in the module.

        dev: Device
            The device we should run this function on.

        number: int
            The number of times to run this function for taking average.
            We call these runs as one `repeat` of measurement.

        repeat: int, optional
            The number of times to repeat the measurement.
            In total, the function will be invoked (1 + number x repeat) times,
            where the first one is warm up and will be discarded.
            The returned result contains `repeat` costs,
            each of which is an average of `number` costs.

        min_repeat_ms: int, optional
            The minimum duration of one `repeat` in milliseconds.
            By default, one `repeat` contains `number` runs. If this parameter is set,
            the parameters `number` will be dynamically adjusted to meet the
            minimum duration requirement of one `repeat`.
            i.e., When the run time of one `repeat` falls below this time, the `number` parameter
            will be automatically increased.

        cooldown_interval_ms: int, optional
            The cooldown interval in milliseconds between the number of repeats defined by
            `repeats_to_cooldown`.

        repeats_to_cooldown: int, optional
            The number of repeats before the cooldown is activated.

        f_preproc: str, optional
            The preprocess function name we want to execute before executing the time evaluator.

        Note
        ----
        The function will be invoked  (1 + number x repeat) times,
        with the first call discarded in case there is lazy initialization.

        Example
        -------
        Normal use with a VM function (may not work over RPC if the function returns a tuple):

        .. code-block:: python

            target = tvm_tirx.target.Target("llvm", host="llvm")
            ex = tvm.compile(TestTimeEvaluator, target)
            vm = relax.VirtualMachine(mod, tvm_tirx.cpu())
            timing_res = vm.time_evaluator("func_name", tvm_tirx.cpu())(arg0, arg1, ..., argn)

        Use with the stateful API:

        .. code-block:: python

            target = tvm_tirx.target.Target("llvm", host="llvm")
            ex = tvm.compile(TestTimeEvaluator, target)
            vm = relax.VirtualMachine(mod, tvm_tirx.cpu())
            vm.set_input("func_name", arg0, arg1, ..., argn)
            timing_res = vm.time_evaluator("invoke_stateful", tvm_tirx.cpu())("func_name")

        With saved closures via `save_function` (this results in
        fewer dictionary lookups in the timed portion):

        .. code-block:: python

            target = tvm_tirx.target.Target("llvm", host="llvm")
            ex = tvm.compile(TestTimeEvaluator, target)
            vm = relax.VirtualMachine(mod, tvm_tirx.cpu())
            vm.save_function("func_name", "func_name_saved", arg0, arg1, ..., argn)
            timing_res = vm.time_evaluator("func_name_saved", tvm_tirx.cpu())()

        Returns
        -------
        ftimer : function
            The function that takes same argument as func and returns a BenchmarkResult.
            The ProfileResult reports `repeat` time costs in seconds.

        """
        return time_evaluator(
            self.module,
            func_name,
            dev,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            cooldown_interval_ms=cooldown_interval_ms,
            repeats_to_cooldown=repeats_to_cooldown,
            f_preproc=f_preproc,
        )


@register_global_func("vm.builtin.debug_print")
def _print(lineo: str, array) -> None:
    print(f"{lineo}: shape = {array.shape}, dtype = {array.dtype}, data =\n{array}")
