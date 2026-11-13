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
"""Performance debug tool for dynamic shape workloads"""

from pathlib import Path

import cloudpickle
import tvm_ffi

import tvm
import tvm_tirx
from tvm import relax

from .utils import default_dym_var_sample_func, get_func_name_from_gv

SKETCH = """import pickle

import tvm, tvm_tirx
from tvm import relax
from tvm_tirx.script import tirx as T

from tvm.s_tir.dlight.benchmark import benchmark_function

MODEL_NAME = "{model_name}"
RELAX_FUNC_NAME = "{relax_func_name}"
FUNCTION_NAME = "{function_name}"
FUNC_HASH = {func_hash}
WEIGHT = {weight}
SAMPLE_NUMBER = {sample_number}

DYM_VAR_SAMPLE_FUNC = {dym_var_sample_func}

# None means extract from Function
INPUT_ARGS = {input_args}
DYM_VAR_DICT = {dym_var_dict}

{func_script}

if __name__ == "__main__":
    target = tvm_tirx.target.Target({target})
    benchmark_function(
        main,
        args = INPUT_ARGS,
        dym_var_dict = DYM_VAR_DICT,
        dym_var_sample_func = DYM_VAR_SAMPLE_FUNC,
        sample_number = SAMPLE_NUMBER,
        target = target,
        weight = WEIGHT,
        relax_func_name = RELAX_FUNC_NAME,
        function_name = FUNCTION_NAME,
    )
"""


def extract_shape(
    arg: tuple | list | relax.Tuple | relax.ShapeType,
) -> list[relax.ShapeType]:
    """Extract shape information from a relax argument.

    Parameters
    ----------
    arg : Union[Tuple, List, relax.Tuple, relax.ShapeType]
        The relax argument to be extracted.

    Returns
    -------
    result : List[relax.ShapeType]
        The extracted shape information.
    """
    if isinstance(arg, tuple | list | tvm.relax.Tuple):
        results = []
        for sub_arg in arg:
            results.extend(extract_shape(sub_arg))
        return results
    return [arg.ty]


def extract_dynamic_var(
    func_dict: dict[
        tvm_tirx.ir.GlobalVar,
        dict[
            tvm_tirx.ir.GlobalVar,
            list[tuple[list, int]],
        ],
    ],
) -> dict[tvm_tirx.ir.GlobalVar, dict[str, str]]:
    """Extract dynamic shape variables from a relax function dictionary.

    Parameters
    ----------
    func_dict : Dict[
        tvm_tirx.ir.GlobalVar,
        Dict[
            tvm_tirx.ir.GlobalVar,
            List[Tuple[List, int]],
        ],
        The relax function dictionary, containing the input arguments' shape information of each
        Function in a Relax function.

    Returns
    -------
    result : Dict[tvm_tirx.ir.GlobalVar, Dict[str, str]]
        The dictionary of dynamic shape variables. Given in format {"n": "int32", "m": "int32"}.
    """
    dym_var_dict: dict[tvm_tirx.ir.GlobalVar, dict[str, str]] = {}
    for gv in func_dict:  # pylint: disable=invalid-name,too-many-nested-blocks
        dym_var_dict[gv] = {}
        for functor in func_dict[gv]:
            for arg_list, _ in func_dict[gv][functor]:
                flattened_arg_list = []
                for arg in arg_list:
                    if isinstance(arg, relax.TupleType):
                        flattened_arg_list.extend(arg.fields)
                    else:
                        flattened_arg_list.append(arg)
                for arg in flattened_arg_list:
                    if isinstance(arg, relax.TensorType):
                        for val in arg.shape.values:
                            if tvm_tirx.ir.is_prim_var(val):
                                dym_var_dict[gv][val.name] = str(val.ty)
                    elif isinstance(arg, relax.ShapeType):
                        for val in arg.values:
                            if tvm_tirx.ir.is_prim_var(val):
                                dym_var_dict[gv][val.name] = str(val.ty)
                    else:
                        raise NotImplementedError
    return dym_var_dict


def update_records(
    records: dict[list[relax.ShapeType], int], new_args: list[relax.ShapeType]
) -> None:
    """Update the count of a function input argument config.

    Parameters
    ----------
    records : Dict[List[relax.ShapeType], int]
        The dictionary to count how many times a function input argument config appears.
    new_args : List[relax.ShapeType]
        The new input argument config.
    """
    for i, (args, count) in enumerate(records):
        if new_args == args:
            records[i] = (args, count + 1)
            return
    records.append((new_args, 1))


def extract_func_info_from_function(
    func: tvm_tirx.tirx.Function,
) -> tuple[list[tuple[tuple[tvm_tirx.tirx.Var | int, ...], str]], dict[str, str]]:
    """Extract function input information from a Function.

    Parameters
    ----------
    func : tvm_tirx.tirx.Function
        The Function to be analyzed.

    Returns
    -------
    result : Tuple[
        List[Tuple[Tuple[Union[tvm_tirx.tirx.Var, int], ...], str]],
        Dict[str, str],
    ]
        The function input information and dynamic shape variable dictionary.
    """
    func_args = []
    dym_var = {}
    for param in func.params:
        if not tvm_tirx.tirx.is_tensor_var(param):
            continue
        tensor = param
        shape = []
        for dim in tensor.shape:
            if isinstance(dim, tvm_tirx.ir.IntImm):
                shape.append(dim.value)
            elif tvm_tirx.ir.is_prim_var(dim):
                dym_var[dim.name] = str(dim.ty)
                shape.append(dim)
            else:
                raise ValueError(f"Unknown shape: {tensor.shape}")
        func_args.append((tuple(shape), str(tensor.dtype)))
    return func_args, dym_var


def extract_all_func_info_from_relax(
    mod: tvm_tirx.ir.IRModule,
) -> tuple[
    dict[tvm_tirx.ir.GlobalVar, dict[tvm_tirx.ir.GlobalVar, list[tuple[list, int]]]],
    dict[tvm_tirx.ir.GlobalVar, dict[str, str]],
]:
    """Extract function input information from a relax module.

    Parameters
    ----------
    mod : tvm_tirx.ir.IRModule
        The Relax module to be analyzed.

    Returns
    -------
    result : Tuple[
        Dict[tvm_tirx.ir.GlobalVar, Dict[tvm_tirx.ir.GlobalVar, List[Tuple[List, int]]]],
        Dict[tvm_tirx.ir.GlobalVar, Dict[str, str]],
    ]
        The function input information and dynamic shape variable dictionary.
    """
    relax_func_dict: dict[
        tvm_tirx.ir.GlobalVar, dict[tvm_tirx.ir.GlobalVar, list[tuple[list, int]]]
    ] = {}
    for gv, func in mod.functions_items():  # pylint: disable=invalid-name,too-many-nested-blocks
        if isinstance(func, tvm.relax.Function):
            for block in func.body.blocks:
                for binding in block.bindings:
                    if isinstance(binding.value, tvm_tirx.ir.Call):
                        raw_args = binding.value.args
                        functor = raw_args[0]
                        if isinstance(functor, tvm_tirx.ir.GlobalVar) and isinstance(
                            mod.functions[functor], tvm_tirx.tirx.Function
                        ):
                            args = extract_shape(raw_args[1:]) + extract_shape(binding.value)
                            if isinstance(functor, tvm_tirx.ir.GlobalVar):
                                if gv not in relax_func_dict:
                                    relax_func_dict[gv] = {}
                                if functor not in relax_func_dict[gv]:
                                    relax_func_dict[gv][functor] = []
                                update_records(relax_func_dict[gv][functor], args)

    return relax_func_dict, extract_dynamic_var(relax_func_dict)


def extract_function(  # pylint: disable=too-many-arguments
    model_name: str,
    relax_func_name: str,
    function_name: str,
    func: tvm_tirx.tirx.Function,
    *,
    func_args: list[tuple[tuple[tvm_tirx.ir.Call | int, ...], str]] | None = None,
    dym_var_dict: dict[str, str] | None = None,
    weight: int = 1,
    sample_number: int = 5,
    target: str | dict | tvm_tirx.target.Target | None = None,
) -> str:
    """Extract a self-contained Function test file from a Relax module.

    Parameters
    ----------
    model_name: str
        The name of the model.
    relax_func_name: str
        The name of the Relax function.
    function_name: str
        The name of the functiontion.
    func: tvm_tirx.tirx.Function
        The Function to be extracted.
    func_args: Optional[List[Tuple[Tuple[Union[tvm_tirx.ir.Call, int], ...], str]]]
        The arguments of the functiontion, including both static and dynamic shape arguments.
        Given in format [ ..., ((1, n, 128), "float32"), ... ].
        If not given, the arguments will be extracted from the Function.
    dym_var_dict: Optional[Dict[str, str]]
        The dictionary of dynamic shape variables. Given in format {"n": "int32", "m": "int32"}.
        If not given, the dictionary will be extracted from the Function.
    weight: int
        The weight of the functiontion, by default 1.
    sample_number: int
        The number of times to sample dynamic shape variables, by default 5.
    target: Optional[Union[str, dict, tvm_tirx.target.Target]]
        The target device to run the Function. Must be supplied explicitly.

    Returns
    -------
    result : str
        The extracted Function test file content.
    """
    if target is None:
        raise ValueError("Target is not specified.")
    elif isinstance(target, str | dict):
        target = tvm_tirx.target.Target(target)
    elif not isinstance(target, tvm_tirx.target.Target):
        raise TypeError("Unsupported target type: " + str(type(target)))
    target_json = str(target)

    return SKETCH.format(
        **{
            "model_name": model_name,
            "relax_func_name": relax_func_name,
            "function_name": function_name,
            "func_hash": tvm_ffi.structural_hash(func),
            "weight": weight,
            "sample_number": sample_number,
            "dym_var_dict": f"pickle.loads({cloudpickle.dumps(dym_var_dict)})"
            if dym_var_dict is not None
            else "None",
            "input_args": f"pickle.loads({cloudpickle.dumps(func_args)})" if func_args else "None",
            "dym_var_sample_func": "pickle.loads("
            + f"{cloudpickle.dumps(default_dym_var_sample_func)}"
            + ")",
            "func_script": func.script(),
            "target": target_json,
        }
    )


def extract_from_relax(
    mod: tvm_tirx.ir.IRModule,
    model_name: str,
    file_path: str,
    target: str | dict | tvm_tirx.target.Target | None = None,
) -> None:
    """Extract self-contained Function test files from a Relax module.

    Parameters
    ----------
    mod: tvm_tirx.ir.IRModule
        The Relax module to be extracted.
    model_name: str
        The name of the model.
    file_path: str
        The path to store the extracted files.
    target: Optional[Union[str, tvm_tirx.target.Target]]
        The target device to run the Function. Must be supplied explicitly.
    """
    relax_funcs, dym_var_dict = extract_all_func_info_from_relax(mod)
    Path(file_path).mkdir(parents=True, exist_ok=True)
    for relax_func_gv in relax_funcs:  # pylint: disable=consider-using-dict-items
        relax_func_name = get_func_name_from_gv(relax_func_gv)
        for function_gv in relax_funcs[relax_func_gv]:
            function_name = get_func_name_from_gv(function_gv)
            for func_args, weight in relax_funcs[relax_func_gv][function_gv]:
                with open(
                    f"{file_path}/{relax_func_name}_{function_name}.py", "w", encoding="utf-8"
                ) as file:
                    print(
                        extract_function(
                            model_name=model_name,
                            relax_func_name=relax_func_name,
                            function_name=function_name,
                            func=mod[function_gv],
                            dym_var_dict=dym_var_dict[relax_func_gv],
                            func_args=func_args,
                            weight=weight,
                            target=target,
                        ),
                        file=file,
                    )
