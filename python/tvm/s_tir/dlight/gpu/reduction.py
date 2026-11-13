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
"""A rule for reduction."""

# TODO: combine reduction rule and general reduction rule into one file.
from collections.abc import Mapping

import tvm_ffi

import tvm_tirx
from tvm import s_tir
from tvm_tirx import sym, tirx
from tvm_tirx.target import Target

from ..analysis import (
    SBlockInfo,
    detect_dominant_read,
    is_broadcast_epilogue,
    normalize_function,
)
from ..base import suggest_threads_per_block, try_inline_contiguous_spatial
from .base import GPUScheduleRule


def _get_reduction_expr(block: s_tir.SBlock) -> tirx.Expr | None:
    # Detect and return `Y` in `X[...] = X[...] + Y`
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


def _has_reduction_loop(block_info):
    return any([info.kind == "R" for info in block_info.iters])


def _suggest_inner_spatial_tx(s_factor: int | tirx.Expr) -> int:
    """Pick the largest divisor no greater than 16."""
    if not isinstance(s_factor, int):
        return 1
    len_tx = 16
    while len_tx > 1 and s_factor % len_tx != 0:
        len_tx -= 1
    return len_tx


def _get_spatial_domains_in_access_order(
    block_info: SBlockInfo, access: sym.IterSumExpr
) -> list[int | tirx.Expr] | None:
    """Return normalized spatial extents in access order."""
    iter_to_info = {info.var: info for info in block_info.iters}
    spatial_domains = []
    seen = set()
    for split_expr in access.args:
        var = split_expr.source.source
        info = iter_to_info.get(var)
        if info is None:
            return None
        if info.kind == "S":
            if var in seen:
                return None
            seen.add(var)
            # `_normalize` fuses the outer loop produced by this split.
            if split_expr.lower_factor > 1:
                extent = split_expr.extent
                extent = int(extent) if isinstance(extent, tvm_tirx.ir.IntImm) else extent
                spatial_domains.append(extent)
            else:
                spatial_domains.append(info.dom)

    # `_normalize` appends omitted unit spatial loops.
    for info in block_info.iters:
        if info.kind == "S" and info.var not in seen:
            if not isinstance(info.dom, int) or info.dom != 1:
                return None
            spatial_domains.append(info.dom)

    if len(spatial_domains) != sum(info.kind == "S" for info in block_info.iters):
        return None
    return spatial_domains


def _inner_spatial_write_back_is_affine(spatial_domains: list[int | tirx.Expr]) -> bool:
    """Check whether the write-back bindings remain quasi-affine."""
    if not spatial_domains:
        return False
    # Each non-unit outer dimension adds a div/mod component to the binding.
    components = sum(1 for dom in spatial_domains[:-1] if not isinstance(dom, int) or dom > 1)
    innermost = spatial_domains[-1]
    if not isinstance(innermost, int) or _suggest_inner_spatial_tx(innermost) < innermost:
        # A partial innermost tile adds one more component.
        components += 1
    return components < 3


class Reduction(GPUScheduleRule):
    """A rule for Reduction."""

    def apply(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None
        sch = s_tir.Schedule(func)
        block_infos = normalize_function(sch)
        if block_infos is None:
            return None
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if len(block_infos) == 1:
            epilogue = None
        elif len(block_infos) == 2:
            epilogue = block_infos[1]
            if not epilogue.is_injective():
                return None
        else:
            return None

        block_info = block_infos[0]
        block = block_info.block_rv
        block_stmt = sch.get(block)

        # Step 1. Check reduction block
        if (
            (not block_info.is_reduction())
            or (not _has_reduction_loop(block_info))
            or len(block_stmt.writes) != 1
            or _get_reduction_expr(block_stmt) is None
        ):
            return None
        # Step 2. Normalize the block, merge spatial and reduction iters
        access = sym.normalize_to_iter_sum(
            detect_dominant_read(block_stmt),
            input_iters={i.var: i.dom for i in block_stmt.iter_vars},
        )
        spatial_domains = _get_spatial_domains_in_access_order(block_info, access)
        if spatial_domains is None:
            return None
        is_inner_reduction, c_factor, loop_order, s_split_index = self._normalize(
            sch, block_info, access
        )
        if is_inner_reduction is None and c_factor is None:
            return None
        # Step 3. Do the scheduling
        if is_inner_reduction:
            self._sch_inner_reduction(
                sch, target, block, c_factor, epilogue, loop_order, s_split_index
            )
        else:
            if not _inner_spatial_write_back_is_affine(spatial_domains):
                return None
            self._sch_inner_spatial(
                sch,
                target,
                block,
                spatial_domains[-1],
                c_factor,
                epilogue,
                loop_order,
                s_split_index,
            )
        return sch

    def _normalize(  # pylint: disable=too-many-branches
        self,
        sch: s_tir.Schedule,
        block_info: SBlockInfo,
        access: sym.IterSumExpr,
    ) -> tuple[bool | None, int | None, Mapping[int, int] | None, int | None]:
        if access.base != 0:
            return None, None, None, None
        iter_to_info = {i.var: i for i in block_info.iters}
        s_loops, r_loops, c_loops, c_factor = [], [], [], None
        s_split_loop, s_split_index = None, None
        for split_expr in access.args:
            var = split_expr.source.source
            info = iter_to_info.pop(var)
            loop = info.loop_rv
            is_inner_reduction = info.kind == "R"
            if split_expr.lower_factor > 1:
                if c_loops:
                    return None, None, None, None
                s_split_loop = loop
                s_split_index = len(s_loops)
                loop, c_loop = sch.split(loop, factors=[None, split_expr.lower_factor])
                c_loops.append(c_loop)
                if not is_inner_reduction:
                    c_factor = split_expr.lower_factor
            if is_inner_reduction:
                r_loops.append(loop)
            else:
                s_loops.append(loop)

        if iter_to_info:
            for var, info in iter_to_info.items():
                if info.kind == "S" and info.dom == 1:
                    s_loops.append(info.loop_rv)
                else:
                    return None, None, None, None

        loop_order = {}
        s_block_var_loops = []
        for i in block_info.iters:
            if i.loop_rv in s_loops or i.loop_rv == s_split_loop:
                s_block_var_loops.append(i.loop_rv)

        for i in range(len(s_block_var_loops)):
            for j in range(len(s_loops)):
                if s_block_var_loops[i] == s_loops[j]:
                    loop_order[i] = j
                    break
                if s_block_var_loops[i] == s_split_loop:
                    loop_order[i] = s_split_index
                    break

        assert s_loops
        assert r_loops
        if len(s_loops) != len([i for i in block_info.iters if i.kind == "S"]):
            return None, None, None, None
        if not c_loops:
            c_loops = [sch.add_unit_loop(block_info.block_rv)]
        sch.reorder(*s_loops, *r_loops, *c_loops)
        sch.fuse(*s_loops)
        sch.fuse(*r_loops)
        return is_inner_reduction, c_factor, loop_order, s_split_index

    def _sch_inner_reduction(  # pylint: disable=too-many-arguments
        self,
        sch: s_tir.Schedule,
        target: Target,
        block: s_tir.schedule.SBlockRV,
        unroll_spatial_factor: int | None,
        epilogue_info: SBlockInfo | None,
        loop_order,
        s_split_index,
    ):
        # pylint: disable=invalid-name
        _, r, _ = sch.get_loops(block)
        (len_tx,) = suggest_threads_per_block(  # pylint: disable=unbalanced-tuple-unpacking
            target, [sch.get(r)]
        )

        _, tx = sch.split(r, factors=[None, len_tx])
        # Schedule the RF block
        rf = sch.rfactor(tx, 0)
        bx, r, tx, _ = sch.get_loops(rf)
        sch.reorder(bx, tx, r)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.annotate(tx, ann_key="tirx.auto_unroll_max_step", ann_val=256)
        sch.annotate(tx, ann_key="tirx.unroll_explicit", ann_val=1)
        sch.set_scope(rf, 0, "local")
        sch.decompose_reduction(rf, r)
        # Schedule the write back block
        sch.reverse_compute_at(block, bx, preserve_unit_loops=True)
        _, tx, *s = sch.get_loops(block)

        if unroll_spatial_factor:
            assert len(s) == len(loop_order)
            new_order_s = [s[loop_order[i]] for i in range(len(s))]
            sch.reorder(*new_order_s)
            new_order_s[s_split_index], c = sch.split(
                new_order_s[s_split_index], factors=[None, unroll_spatial_factor]
            )
            sch.reorder(*new_order_s, c)
            s = sch.fuse(*new_order_s)
            sch.reorder(s, tx, c)
        else:
            s = sch.fuse(*s)
            sch.reorder(s, tx)
        sch.bind(tx, "threadIdx.x")
        # Schedule epilogue
        if epilogue_info is not None:
            epilogue = epilogue_info.block_rv
            sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
            if is_broadcast_epilogue(sch, block, epilogue):
                sch.set_scope(block, 0, "shared")
                _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                _, tx = sch.split(sch.fuse(*s), factors=[None, len_tx])
                sch.bind(tx, "threadIdx.x")
            else:
                sch.set_scope(block, 0, "local")
        # pylint: enable=invalid-name

    def _sch_inner_spatial(
        self,
        sch: s_tir.Schedule,
        _: Target,
        block: s_tir.schedule.SBlockRV,
        s_factor: int | tirx.Expr,
        unroll_spatial_factor: int | None,
        epilogue_info: SBlockInfo | None,
        loop_order,
        s_split_index,
    ):
        # pylint: disable=invalid-name
        s, r, _ = sch.get_loops(block)
        len_ty = 16
        # get perfect spatial factor, spatial factor should be divide the innermost spatial loop so
        # that the block after r_factor and be reversed compute at the original scope
        len_tx = _suggest_inner_spatial_tx(s_factor)
        _, _ = sch.split(s, factors=[None, len_tx])
        _, ty = sch.split(r, factors=[None, len_ty])
        # Schedule the RF block
        rf = sch.rfactor(ty, 0)
        bx, tx, r, ty, _ = sch.get_loops(rf)
        sch.reorder(bx, tx, ty, r)
        sch.bind(tx, "threadIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.set_scope(rf, 0, "local")
        sch.decompose_reduction(rf, r)
        # Schedule the write back block
        sch.reverse_compute_at(block, bx, preserve_unit_loops=True)
        _, r, *s = sch.get_loops(block)
        if unroll_spatial_factor:
            assert len(s) == len(loop_order)
            new_order_s = [s[loop_order[i]] for i in range(len(s))]
            sch.reorder(*new_order_s)
            new_order_s[s_split_index], c = sch.split(
                new_order_s[s_split_index], factors=[None, unroll_spatial_factor]
            )
            sch.reorder(*new_order_s, c)
            s = sch.fuse(*new_order_s)
            sch.reorder(s, c, r)
        else:
            s = sch.fuse(*s)
            sch.reorder(s, r)
        sch.bind(s, "threadIdx.x")
        sch.bind(r, "threadIdx.y")

        # Schedule epilogue
        if epilogue_info is not None:
            epilogue = epilogue_info.block_rv
            sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
            if is_broadcast_epilogue(sch, block, epilogue):
                sch.set_scope(block, 0, "shared")
                _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                _, tx, ty = sch.split(sch.fuse(*s), factors=[None, len_tx, len_ty])
                sch.bind(tx, "threadIdx.x")
                sch.bind(ty, "threadIdx.y")
            else:
                # The epilogue is element-wise without broadcasting.
                # Thus the remaining spatial part should be bind to tx.
                sch.set_scope(block, 0, "local")
                _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                tx, _ = sch.split(sch.fuse(*s), factors=[len_tx, None])
                sch.bind(tx, "threadIdx.x")
        # pylint: enable=invalid-name
