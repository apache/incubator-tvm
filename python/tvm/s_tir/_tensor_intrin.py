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
"""Tensor intrinsics for schedulable TIR."""

from typing import Optional

import tvm_ffi

from tvm_tirx.tirx.function import Function

from . import _ffi_api


@tvm_ffi.register_object("s_tir.TensorIntrin")
class TensorIntrin(tvm_ffi.Object):
    """A tensor intrinsic.

    Parameters
    ----------
    desc : Function
        The function to describe the computation.

    impl : Function
        The function of the implementation for the execution.
    """

    def __init__(self, desc, impl):
        self.__init_handle_by_constructor__(_ffi_api.TensorIntrin, desc, impl)

    @staticmethod
    def register(name: str, desc: Function, impl: Function, override: bool = False):
        """Register a tensor intrinsic with its name.

        Parameters
        ----------
        name : str
            The name of the TensorIntrin to register.
        desc : Function
            The function to describe the computation.
        impl : Function
            The function of the implementation for the execution.
        override: bool
            Whether override existing intrinsic.
        """
        return _ffi_api.TensorIntrinRegister(name, TensorIntrin(desc, impl), override)  # type: ignore

    @staticmethod
    def get(name: str, allow_missing: bool = False) -> Optional["TensorIntrin"]:
        """Look up a tensor intrinsic by its name.

        Parameters
        ----------
        name : str
            The name of the TensorIntrin to look up.

        allow_missing : bool
            Whether to allow missing tensor intrin. If False, raise an error if the tensor intrin
        doesn't exist.

        Returns
        -------
        result : Optional[TensorIntrin]
            The TensorIntrin with the specified name, or None if not found.
        """
        return _ffi_api.TensorIntrinGet(name, allow_missing)  # pylint: type: ignore
