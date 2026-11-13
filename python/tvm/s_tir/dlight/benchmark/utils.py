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
"""Util functions for benchmarking dynamic shape workloads"""

from typing import Any

import tvm_tirx
from tvm import relax

INPUT_SHAPE_TYPE = list[tuple[tuple[int, ...], str]]  # pylint: disable=invalid-name


def _dtype_str(dtype) -> str:
    if isinstance(dtype, tvm_tirx.ir.PrimType):
        dtype = dtype.dtype
    return str(dtype)


def get_func_name_from_gv(gv: tvm_tirx.ir.GlobalVar) -> str:  # pylint: disable=invalid-name
    """Get function name from a global variable.

    Parameters
    ----------
    gv : tvm_tirx.ir.GlobalVar
        The given global variable.

    Returns
    -------
    result : str
        The global variable name without the prefix "...@".
    """
    return gv.name_hint


def dym_var_sample_str(sample: dict[str | tvm_tirx.ir.Call, int]) -> str:
    """Convert a variable value sample to a string.

    Parameters
    ----------
    sample : Dict[Union[str, tvm_tirx.ir.Call], int]
        Variable value sample, e.g., {n: 64, m: 128} or {"n": 64, "m": 128}

    Returns
    -------
    result : str
        Variable value sample string, e.g., "n=64, m=128"
    """
    return ", ".join([f"{k}={v}" for k, v in sample.items()])


def populuate_input_shape(
    input_infos: list[relax.TensorType | tuple[tuple[int | str, ...], str]],
    dym_var_sample: dict[str, int],
) -> INPUT_SHAPE_TYPE:
    """
    Populate input shapes with dynamic shape variable samples.

    Parameters
    ----------
    input_infos : List[Union[relax.TensorType, Tuple[Tuple[Union[int, str], ...], str]]]
        Input tensor information, including shape and dtype,
        e.g., [..., Shape(1, n, 128) with dtype="int32", ...]
    dym_var_sample : Dict[str, int]
        Dynamic shape variable sample, e.g., {"n": 64}

    Returns
    -------
    results : INPUT_SHAPE_TYPE
        Input shapes with dynamic shape variable samples, e.g.,
        [..., ((1, 64, 128), "int32"), ...] if n=64 or
        [..., (128, "scalar"), ...] if n=128 for scalar input
    """
    results: INPUT_SHAPE_TYPE = []
    for input_info in input_infos:
        shape = []
        if isinstance(input_info, relax.ShapeType):
            # scalar input
            value = input_info.values[0]
            key = value.name if isinstance(value, tvm_tirx.ir.Var) else str(value)
            results.append(((dym_var_sample[key],), "scalar"))
        else:
            if isinstance(input_info, relax.TensorType):
                tensor_shape = input_info.shape
                tensor_dtype = input_info.dtype
            else:
                tensor_shape, tensor_dtype = input_info  # type: ignore
            for dim in tensor_shape:
                if isinstance(dim, int):
                    shape.append(dim)
                elif isinstance(dim, tvm_tirx.ir.IntImm):
                    shape.append(dim.value)
                else:
                    key = dim.name if isinstance(dim, tvm_tirx.ir.Var) else str(dim)
                    shape.append(dym_var_sample[key])
            results.append(((*shape,), _dtype_str(tensor_dtype)))
    return results


def default_dym_var_sample_func(dym_var_dict: dict[str, str]) -> dict[str, int]:
    """
    Default dynamic shape variable sample function.
    Sample a random value for each dynamic shape variable.

    Parameters
    ----------
    dym_var_dict : Dict[str, str]
        Dynamic shape variable dictionary, e.g., {"n": "int32", "m": "int32"}

    Returns
    -------
    result : Dict[str, int]
        Dynamic shape variable sample, e.g., {"n": 64, "m": 128}
    """
    results = {}
    for var in dym_var_dict:
        if dym_var_dict[var] in ["int32", "int64"]:
            import random  # pylint: disable=import-outside-toplevel

            results[var] = random.randint(2, 128)
        else:
            raise TypeError("Unsupported dynamic shape variable type: " + dym_var_dict[var])
    return results


def print_results(
    bench_results: list[dict[str, Any]], sort_by: str = "WxTime(ms)", desc: bool = True
):
    """Print benchmark results.

    Parameters
    ----------
    bench_results : List[Dict[str, Any]]
        Benchmark results as dictionary list.
    sort_by : str
        Sort results by this key, if None, no sorting.
    desc : bool
        Whether to sort results in descending order.
    """
    # pylint: disable=invalid-name, import-outside-toplevel
    try:
        import pandas as pd

        df = pd.DataFrame()
        for record in bench_results:
            df = pd.concat(
                [df, pd.DataFrame(record, index=[0])],
                ignore_index=True,
            )
        if sort_by is not None:
            if sort_by not in df.columns:
                raise ValueError(f"sort_by key {sort_by} not in benchmark results")
            df = df.sort_values(sort_by, ascending=not desc).reset_index().drop("index", axis=1)
        print(df)
    except ModuleNotFoundError:
        print("Pandas not found, printing results in raw format.")
        keys = []
        if len(bench_results) > 0:
            for key in bench_results[0]:
                keys.append(str(key))
        print("\t".join(keys))
        for record in bench_results:
            values = []
            for key in keys:
                values.append(str(record[key]))
            print("\t".join(values))
    print("\n")
    # pylint: enable=invalid-name, import-outside-toplevel
