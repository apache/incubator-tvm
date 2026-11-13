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
"""VM parameter dictionary serialization."""

from tvm_ffi import get_global_func

from tvm_tirx.runtime import Tensor, tensor


def _to_tensor(params):
    transformed = {}

    for k, v in params.items():
        if not isinstance(v, Tensor):
            transformed[k] = tensor(v)
        else:
            transformed[k] = v

    return transformed


def save_param_dict(params):
    """Save parameter dictionary to binary bytes.

    The resulting bytes can be loaded with :func:`load_param_dict`.

    Parameters
    ----------
    params : dict of str to Tensor
        The parameter dictionary.

    Returns
    -------
    param_bytes: bytearray
        Serialized parameters.

    Examples
    --------
    .. code-block:: python

       # set up the parameter dict
       params = {"param0": arr0, "param1": arr1}
       # save the parameters as byte array
       param_bytes = tvm.runtime.vm.save_param_dict(params)
       # We can serialize the param_bytes and load it back later.
       # Recover the parameter dictionary from the serialized bytes.
       tvm.runtime.vm.load_param_dict(param_bytes)
    """
    return get_global_func("vm.SaveParams")(_to_tensor(params))


def save_param_dict_to_file(params, path):
    """Save parameter dictionary to file.

    Parameters
    ----------
    params : dict of str to Tensor
        The parameter dictionary.

    path: str
        The path to the parameter file.
    """
    return get_global_func("vm.SaveParamsToFile")(_to_tensor(params), path)


def load_param_dict(param_bytes):
    """Load parameter dictionary from binary bytes.

    Parameters
    ----------
    param_bytes: bytearray
        Serialized parameters.

    Returns
    -------
    params : dict of str to Tensor
        The parameter dictionary.
    """
    if isinstance(param_bytes, bytes | str):
        param_bytes = bytearray(param_bytes)
    return get_global_func("vm.LoadParams")(param_bytes)


def load_param_dict_from_file(path):
    """Load parameter dictionary from file.

    Parameters
    ----------
    path: str
        The path to the parameter file to load from.

    Returns
    -------
    params : dict of str to Tensor
        The parameter dictionary.
    """
    return get_global_func("vm.LoadParamsFromFile")(path)
