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
# pylint: disable=unused-argument
"""Tools to compare libraries."""

from collections.abc import Iterable

import tvm_tirx
import tvm_tirx.testing


class LibCompareVMInstrument:
    """Instrument class to compare libs.

    This class build an instrument function that
    pair tests an existing compiled relax vm implementation
    and an extra module, which can sits in another backend
    but offers a same subset of compiled TIR functions.

    The instrumentation enables us to automatically
    check and compare each ops being called in the pipeline
    by looking up the same name in the provided mod and run testing.

    Parameters
    ----------
    mod: tvm_ffi.Module
        The module of interest to be validated.

    device: runtime.Device
        The device to run the target module on.

    verbose: bool
        Whether print out messages.

    rtol: float
        rtol used in validation

    atol: float
        atol used in validation
    """

    def __init__(self, mod, device, verbose=True, rtol=1e-5, atol=1e-5):
        self.mod = mod
        self.device = device
        self.verbose = verbose
        self.counter = 0
        self.rtol = rtol
        self.atol = atol

    def compare(
        self,
        name: str,
        ref_args: list[tvm_tirx.runtime.Tensor] | tuple[tvm_tirx.runtime.Tensor, ...],
        new_args: list[tvm_tirx.runtime.Tensor] | tuple[tvm_tirx.runtime.Tensor, ...],
        ret_indices: Iterable[int],
    ):
        """Comparison function, can be overloaded.

        Parameters
        ----------
        name: str
            Name of the function.

        ref_args:
            The reference arguments.

        new_args:
            The args to be passed to the comparison function.

        ret_indices:
            List of indices to validate return values.
        """
        my_func = self.mod.get_function(name, query_imports=True)
        if self.verbose:
            print(f"[{self.counter}] Validating {name} ...")
        my_func(*new_args)
        for rindex in ret_indices:
            tvm_tirx.testing.assert_allclose(
                new_args[rindex].numpy(), ref_args[rindex].numpy(), atol=self.atol, rtol=self.rtol
            )
        if self.verbose:
            print(f"[{self.counter}] Validating {name}, passed.")
        self.counter += 1

    def skip_instrument(self, func, name, before_run, ret_val, *args):
        return False

    def __call__(self, func, name, before_run, ret_val, *args):
        if before_run:
            return
        if name.startswith("vm.builtin."):
            return
        if any(not isinstance(x, tvm_tirx.runtime.Tensor) for x in args):
            return
        try:
            self.mod.get_function(name, query_imports=True)
        except AttributeError:
            if self.verbose:
                print(f"Cannot find {name}, skip...")
            return

        if self.skip_instrument(func, name, before_run, ret_val, *args):
            return

        new_args = []
        # not always true, true for most ops.
        ret_indices = (len(args) - 1,)
        temp_args = []
        for i, arg in enumerate(args):
            arr = tvm_tirx.runtime.empty(arg.shape, arg.dtype, device=self.device)
            # copy from cpu since we look at different device
            if i not in ret_indices:
                temp_cpu = arg.copyto(tvm_tirx.cpu())
                temp_args.append(temp_cpu)
                arr.copyfrom(temp_cpu)
            new_args.append(arr)
        # wait until all copy complete before we release temp_cpu
        self.device.sync()
        self.compare(name, args, new_args, ret_indices)
