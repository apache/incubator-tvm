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
"""Analysis for GEMV."""

import tvm_ffi

import tvm_tirx
from tvm import s_tir
from tvm_tirx import sym, tirx

from .common_analysis import (
    SBlockInfo,
    collect_block_iter_vars_used_in_access_region,
    collect_vars_used_in_prim_expr,
    detect_dominant_read,
)


def get_reduction_expr(block: s_tir.SBlock) -> tirx.Expr | None:
    """Extracts the reduction expression from a TIR block.

    This function checks whether the given TIR block follows a reduction pattern
    of the form `X[...] = X[...] + Y` and returns `Y` as the reduction expression.

    Parameters:
    ----------
    block : s_tir.SBlock
        The TIR block to analyze.

    Returns:
    -------
    Optional[tirx.Expr]
        The reduction expression (`Y`) if detected, otherwise None.
    """

    if len(block.body) != 1:
        return None
    tensor_store = block.body[0]
    if not isinstance(tensor_store, tvm_tirx.ir.TensorStore):
        return None
    if not isinstance(tensor_store.value, tirx.Add):
        return None
    if not tvm_ffi.structural_equal(
        tensor_store.value.a,
        tvm_tirx.ir.TensorLoad(tensor_store.dest, tensor_store.indices),
        map_free_vars=True,
    ):
        return None
    return tensor_store.value.b


def is_gemv(sch: s_tir.Schedule, block_info: SBlockInfo) -> list[tirx.Var] | None:
    """Check if the block is a GEMV.

    Parameters
    ----------

    sch : s_tir.Schedule
        The schedule

    block_info : SBlockInfo
        The block info to be checked


    Returns
    -------
    ret : Optional[List[tirx.Var]]
        The vector tensors used in the GEMV if it is a GEMV, otherwise None.
    """
    block = block_info.block_rv
    block_stmt = sch.get(block)
    conditions = []
    conditions.append(block_info.is_reduction())
    conditions.append(len(block_stmt.reads) >= 2)
    conditions.append(len(block_stmt.writes) == 1)
    conditions.append(get_reduction_expr(block_stmt) is not None)
    conditions.append(
        len(collect_block_iter_vars_used_in_access_region(block_stmt, block_stmt.writes[0].region))
        > 0
    )
    if not all(conditions):
        return None

    iter_num = len(block_stmt.iter_vars)
    ret = [
        read.source
        for read in block_stmt.reads
        if len(collect_block_iter_vars_used_in_access_region(block_stmt, read.region)) < iter_num
        and len(collect_block_iter_vars_used_in_access_region(block_stmt, read.region)) > 0
    ]
    return ret if 0 < len(ret) < len(block_stmt.reads) else None


def normalize(
    sch: s_tir.Schedule,
    block_info: SBlockInfo,
) -> bool | None:
    """Normalize the main block."""
    block_stmt: s_tir.SBlock = sch.get(block_info.block_rv)
    access = sym.normalize_to_iter_sum(
        detect_dominant_read(block_stmt),
        input_iters={i.var: i.dom for i in block_stmt.iter_vars},
    )
    tensors_use_vars = [
        collect_block_iter_vars_used_in_access_region(block_stmt, tensor.region)
        for tensor in block_stmt.writes
    ]
    tensors_use_vars.extend(
        [
            collect_block_iter_vars_used_in_access_region(block_stmt, tensor.region)
            for tensor in block_stmt.reads
        ]
    )
    if collect_vars_used_in_prim_expr(access.base) & set(
        iter_var.var for iter_var in block_stmt.iter_vars
    ):
        return None
    iter_to_info = {i.var: i for i in block_info.iters}
    if not access.args or any(
        split_expr.source.source not in iter_to_info for split_expr in access.args
    ):
        return None
    batch_loops, s_loops, r_loops, c_loops = [], [], [], []
    inner_axis = access.args[-1].source.source
    is_inner_reduction = iter_to_info[inner_axis].kind == "R"

    for split_expr in access.args:
        var = split_expr.source.source
        info = iter_to_info.get(var)
        loop = info.loop_rv
        is_reduction = info.kind == "R"
        if split_expr.lower_factor > 1:
            if c_loops:
                return None
            loop, c_loop = sch.split(loop, factors=[None, split_expr.lower_factor])
            # we only support the reduction dim being grouped atm
            if not is_reduction:
                return None
            c_loops.append(c_loop)
        if is_reduction:
            r_loops.append(loop)
        elif all([var in tensor_vars for tensor_vars in tensors_use_vars]):
            batch_loops.append(loop)
        else:
            s_loops.append(loop)

    assert s_loops
    assert r_loops
    if not c_loops:
        c_loops = [sch.add_unit_loop(block_info.block_rv)]
    if not batch_loops:
        batch_loops = [sch.add_unit_loop(block_info.block_rv)]
    sch.reorder(*batch_loops, *s_loops, *r_loops, *c_loops)
    sch.fuse(*batch_loops)
    sch.fuse(*s_loops)
    sch.fuse(*r_loops)
    return is_inner_reduction
