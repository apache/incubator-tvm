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
# pylint: disable=unused-argument, invalid-name, no-else-return, abstract-method, arguments-differ
"""Relax transformation passes for testing"""

import logging
import os

import tvm_ffi

import tvm_tirx
from tvm.relax.expr import DataflowBlock, Var
from tvm_tirx.ir import Call
from tvm_tirx.ir.module import IRModule


def ApplyEmptyCppMutator() -> tvm_tirx.ir.transform.Pass:
    """Create empty cpp mutator"""
    packed_func = tvm_ffi.get_global_func("relax.testing.transform.ApplyEmptyCppMutator")
    return packed_func()


def dataflow_liveness_analysis(block: DataflowBlock) -> dict[Var, tuple[int, int]]:
    """
    Inner function for the dataflow inplace transformation exposed for testing.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.warning("The function dataflow_liveness_analysis is exposed for testing only.")

    live_ranges = tvm_ffi.get_global_func("relax.testing.transform.DataflowLivenessAnalysis")(block)  # type: ignore
    ret = {}
    for var, live_range in live_ranges.items():
        ret[var] = tuple(live_range)
    return ret  # type: ignore


def dataflow_alias_analysis(
    block: DataflowBlock, inputs: list[Var]
) -> tuple[dict[Var, set[int]], dict[int, list[set[int]]]]:
    """
    Inner function for the dataflow inplace transformation exposed for testing.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.warning("The function dataflow_alias_analysis is exposed for testing only.")

    alias_sets, tuple_map = tvm_ffi.get_global_func(
        "relax.testing.transform.DataflowAliasAnalysis"
    )(
        block,
        inputs,
    )  # type: ignore
    res_alias_sets = {}
    res_tuple_map = {}
    for var, alias_set in alias_sets.items():
        res_alias_sets[var] = set(alias_set)
    for idx, elem_alias_sets in tuple_map.items():
        res_tuple_map[idx] = [set(alias_set) for alias_set in elem_alias_sets]
    return res_alias_sets, res_tuple_map  # type: ignore


@tvm_ffi.register_object("relax.transform.InplaceOpportunity")
class InplaceOpportunity(tvm_ffi.Object):
    """
    Represents an opportunity to make a binding in-place. Exposed only for testing;
    the constructor is not exposed.

    Parameters:
    -----------
    binding_idx: int
        Index of the binding within its block

    arg_idxs: List[int]
        Indices of arguments that are eligible to be used as in-place targets.
    """

    def __init__(self, _binding_idx, _arg_idxs):
        raise NotImplementedError("Constructor for InplaceOpportunity not exposed!")


def dataflow_inplace_analysis(
    block: DataflowBlock, inputs: list[Var], mod: IRModule
) -> tuple[list[tuple[int, set[int]]], list[tuple[int, set[int]]]]:
    """
    Inner function for the dataflow inplace transformation exposed for testing.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.warning("The function dataflow_inplace_analysis is exposed for testing only.")
    index_lists = tvm_ffi.get_global_func("relax.testing.transform.DataflowInplaceAnalysis")(
        block, inputs, mod
    )  # type: ignore

    def convert(opp_list):
        return list(map(lambda opp: (int(opp.binding_idx), set(map(int, opp.arg_idxs))), opp_list))

    return (convert(index_lists[0]), convert(index_lists[1]))  # type: ignore


def dataflow_single_inplace_call(
    mod: IRModule, call: Call, inplace_indices: list[int]
) -> tuple[Call, IRModule]:
    """
    Inner function for the dataflow inplace transformation exposed for testing.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.warning("The function dataflow_single_inplace_call is exposed for testing only.")

    ret = tvm_ffi.get_global_func("relax.testing.transform.SingleInplaceCall")(
        mod,
        call,
        inplace_indices,
    )  # type: ignore
    return (ret[0], ret[1])  # type: ignore
