# isort: skip_file
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
"""Analysis utilities for Schedulable TensorIR (S-TIR)."""

# pylint: disable=invalid-name
from typing import Optional, Union

import tvm
import tvm_tirx
from tvm_tirx.ir import IRModule, TensorRegion
from tvm_tirx.tirx.expr import Var
from tvm.s_tir import SBlock

from tvm_tirx.ir import Stmt
from tvm_tirx.tirx.function import Function
from . import _ffi_api


def get_sblock_access_region(
    block: SBlock, tensor_var_map: dict[Var, Var]
) -> list[list[TensorRegion]]:
    """Detect which regions of tensors in this block are read or written to.
       Regions are sorted by order of appearance in the AST.

    Parameters
    ----------
    block: tvm.s_tir.SBlock
        The block in which we are detecting read/write regions.

    tensor_var_map : Dict[Var, Var]
        The outside tensors which may access the block. Mapping from tensor var to the tensor

    Returns
    -------
    result : List[List[TensorRegion]]
        Array of access regions. There are three arrays of TensorRegion:
            - first: read regions
            - second: write regions
            - third: opaque regions
    """
    return _ffi_api.GetSBlockAccessRegion(block, tensor_var_map)  # type: ignore


def get_sblock_read_write_region(
    block: SBlock, tensor_var_map: dict[Var, Var]
) -> list[list[TensorRegion]]:
    """Auto detect the block read/write region according to its body stmt.
       An opaque access will be counted as both a read and a write access

    Parameters
    ----------
    block: tvm.s_tir.SBlock
        The block in which we are detecting read/write regions.

    tensor_var_map : Dict[Var, Var]
        The outside tensors which may access the block. Mapping from tensor var to the tensor

    Returns
    -------
    result : List[List[TensorRegion]]
        An array only consisting of the read regions and write regions of the input block
    """
    return _ffi_api.GetSBlockReadWriteRegion(block, tensor_var_map)  # type: ignore


def detect_tensor_access_lca(func: Function) -> dict[Var, Stmt]:
    """Detect the lowest common ancestor(LCA) of tensor access, including both high-level
    access (TensorLoad, TensorStore) and low-level access (TensorLoad, TensorStore and opaque
    access).
    The LCA may be a For loop or a Block.

    Parameters
    ----------
    func: tvm_tirx.tirx.Function
        The function to be detected.

    Returns
    -------
    result : Dict[Var, Stmt]
        Map from tensor to the LCA of all access to it.
    """
    return _ffi_api.detect_tensor_access_lca(func)  # type: ignore # pylint: disable=no-member


def find_anchor_sblock(mod: IRModule) -> SBlock | None:
    """Find the "anchor block" of the given module.

    We define the anchor block to be the block with (1) an init statement and (2) having
    the biggest flops count. The latter condition is only used when there are multiple blocks
    with an init statement.

    For example, if the input module is conv2d + fused spatial blocks, conv2d is the anchor block.
    The input module may not contain more than one such block. For example, a module having
    two conv2d is not allowed as an input.

    However, a module created from winograd convolution has multiple blocks with an init statement
    (input transform, batched GEMM, and output transform). We use the second condition, the flops
    count, to determine that the batched GEMM block is the anchor block.

    Parameters
    ----------
    mod: tvm_tirx.ir.IRModule
        The input TIR module.
    Returns
    -------
    anchor_block: Optional[SBlock]
        The anchor block if found, None otherwise.
    """
    return _ffi_api.find_anchor_sblock(mod)  # type: ignore # pylint: disable=no-member


def verify_gpu_code(func: Function, constraints: dict[str, int]) -> bool:
    """Verify if module contains illegal host side direct memory access.

    Parameters
    ----------
    func: tvm_tirx.tirx.Function
        The module to be verified.

    constraints : Dict[str, int]
        The attribute constraints.

    Returns
    -------
    result : bool
        The result of verification.
    """
    return _ffi_api.verify_gpu_code(func, constraints)  # type: ignore


def calculate_allocated_bytes(
    func_or_mod: Function | IRModule,
) -> dict[str, dict[str, int]]:
    """Calculate allocated memory per memory scope required by TIR Functions.

    Parameters
    ----------
    func_or_mod: Union[Function, IRModule]
        The function or module to be detected. If a module is passed, allocated
        memory is calculated for all Functions inside the module

    Returns
    -------
    result : Dict[str, Dict[str, int]]
        Allocated memory size per scope in bytes for each function in the IRModule returned as a
        dict with function names as keys and a dict of allocated sizes as values. If a single
        Function is passed, the function name is returned as "main"
    """
    if not isinstance(func_or_mod, Function | IRModule):
        raise TypeError(
            f"Expected argument to be Function or IRModule, but received {type(func_or_mod)}"
        )
    return _ffi_api.calculate_allocated_bytes(func_or_mod)  # type: ignore


def estimate_tir_flops(stmt_or_mod: Stmt | IRModule) -> float:
    """Estimate the FLOPs of a TIR fragment.

    Parameters
    ----------
    stmt_or_mod: Union[Stmt, IRModule]
        The TIR fragment or IRModule to be estimated.

    Returns
    -------
    flops: float
        The estimated FLOPs.
    """
    return _ffi_api.EstimateTIRFlops(stmt_or_mod)  # type: ignore # pylint: disable=no-member


def OOBChecker():
    """Detect out of bounds memory access in arrays.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.OOBChecker()  # type: ignore


def get_vtcm_compaction_passes() -> list[tvm_tirx.ir.transform.Pass]:
    """Utility function to get the list of lowering passes to be applied to calculate the compacted
    VTCM allocation size

    Returns
    -------
    result : List[tvm_tirx.ir.transform.Pass]
        returns list of passes
    """
    return _ffi_api.get_vtcm_compaction_passes()  # type: ignore # pylint: disable=no-member


def is_pure_function(func: Function) -> bool:
    """Checks if the function is a pure function"""
    return _ffi_api.is_pure_function(func, False)  # type: ignore # pylint: disable=no-member


def assert_pure_function(func: Function) -> bool:
    """Asserts that the function is a pure function"""
    return _ffi_api.is_pure_function(func, True)  # type: ignore # pylint: disable=no-member


def check_well_formed(obj: Function | IRModule) -> None:
    """Check whether an S-TIR function or module is well formed.

    Parameters
    ----------
    obj : Function or IRModule
        The function or module to check.

    Raises
    ------
    RuntimeError
        If the function or module is not well formed.
    """
    _ffi_api.CheckWellFormed(obj)
