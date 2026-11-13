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

# pylint: disable=invalid-name,unnecessary-comprehension
"""Testing functions for the RPC server."""

import numpy as np
import tvm_ffi

import tvm_tirx


# RPC test functions to be registered for unit-tests purposes
@tvm_ffi.register_global_func("rpc.test.addone")
def _addone(x):
    return x + 1


@tvm_ffi.register_global_func("rpc.test.strcat")
def _strcat(name, x):
    return f"{name}:{x}"


@tvm_ffi.register_global_func("rpc.test.except")
def _remotethrow(name):
    raise ValueError(f"{name}")


@tvm_ffi.register_global_func("rpc.test.runtime_str_concat")
def _strcat(x, y):
    return x + y


@tvm_ffi.register_global_func("rpc.test.remote_tensor_func")
def _remote_tensor_func(y):
    x = np.ones((3, 4))
    np.testing.assert_equal(y.numpy(), x)


@tvm_ffi.register_global_func("rpc.test.add_to_lhs")
def _add_to_lhs(x):
    return lambda y: x + y


@tvm_ffi.register_global_func("rpc.test.remote_return_nd")
def _my_module(name):
    # Use closure to check the ref counter correctness
    nd = tvm_tirx.runtime.tensor(np.zeros(10).astype("float32"))

    if name == "get_arr":
        return lambda: nd
    if name == "ref_count":
        # Imported lazily: rpc.server imports this module to register the
        # rpc.test.* helpers, so a top-level ``import tvm.testing`` would drag
        # tvm.testing (which imports pytest) into the plain ``import tvm`` path.
        from tvm_tirx.testing import object_use_count

        return lambda: object_use_count(nd)
    if name == "get_elem":
        return lambda idx: nd.numpy()[idx]
    if name == "get_arr_elem":
        return lambda arr, idx: arr.numpy()[idx]
    raise RuntimeError("unknown name")
