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
"""Testing utility functions in meta schedule"""

from collections.abc import Callable

import numpy as np  # type: ignore
import tvm_ffi

import tvm_tirx
from tvm_tirx.runtime import Tensor


def generate_input_data(
    input_shape: list[int],
    input_dtype: str,
    *,
    low: int | None = None,
    high: int | None = None,
) -> np.ndarray:
    """Generate input date with given shape and data type.

    Parameters
    ----------
    input_shape : List[int]
        The shape of the input data.
    input_dtype : str
        The data type of the input date.

    Returns
    -------
    input_data : np.ndarray
        The generated input data with given shape and data type in numpy ndarray.
    """
    if input_dtype.startswith("float"):
        return np.random.uniform(size=input_shape).astype(input_dtype)
    range_map = {
        "uint8": (0, 255),
        "int8": (-128, 127),
        "int32": (0, 10000),
        "uint32": (0, 10000),
        "int64": (0, 10000),
        "uint64": (0, 10000),
    }
    if input_dtype in range_map:
        _low, _high = range_map[input_dtype]
        return np.random.randint(
            low=_low if low is None else low,
            high=_high if high is None else high,
            size=input_shape,
            dtype=input_dtype,
        )
    raise ValueError("Unsupported input datatype!")


def create_calculator(backend: str) -> Callable:
    """Create a function to fetch the computing result of running the given runtime module.

    Parameters
    ----------
    backend : str
        The backend to use, only tirx is supported for now.

    Returns
    -------
    func : Callable
        The function to fetch the computing result.
    """

    def f_calculator(
        rt_mod: tvm_ffi.Module,
        dev: tvm_tirx.runtime.Device,  # pylint: disable=unused-argument
        input_data: dict[str, Tensor],
    ) -> list[Tensor]:
        """Fetch the result of running the given runtime module.

        Parameters
        ----------
        rt_mod : tvm_ffi.Module
            The runtime module.
        dev : tvm_ffi.device
            The device type to run workload.
        input_data : Dict[str, np.ndarray]
            The input data as a dictionary.
        """
        try:
            if backend == "tirx":
                data = [v for _, v in sorted(input_data.items(), key=lambda x: x[0])]
                rt_mod(*data)
                return data
            else:
                raise ValueError(f"Backend {backend} not supported in f_calculator!")

        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"Run module f_calculator via RPC failed, exception: {exc}",
            )
            return None

    return f_calculator
