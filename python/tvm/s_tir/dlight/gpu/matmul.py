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
# pylint: disable=missing-docstring, invalid-name
# ruff: noqa: E722
"""A GEMM schedule rule for GPU operators."""

from dataclasses import dataclass
from enum import Enum

import tvm_tirx
from tvm import s_tir
from tvm.s_tir import IterVar
from tvm.s_tir.schedule.schedule import SBlockRV
from tvm_tirx import tirx
from tvm_tirx.ir import Range
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target
from tvm_tirx.tirx import Expr, Var
from tvm_tirx.tirx.analysis import undefined_vars

from ..analysis import IterInfo, SBlockInfo, get_root_block
from .base import GPUScheduleRule


def _collect_producers(sch: s_tir.Schedule, block: s_tir.schedule.SBlockRV):
    result = []
    for producer in sch.get_producers(block):
        result.append(producer)
        result.extend(_collect_producers(sch, producer))
    return result


def _collect_consumers(sch: s_tir.Schedule, block: s_tir.schedule.SBlockRV):
    result = []
    for consumer in sch.get_consumers(block):
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer))
    return result


def auto_inline_producers(
    sch: s_tir.Schedule,
    block: s_tir.schedule.SBlockRV,
):
    while True:
        inlined_cnt = 0
        producers = _collect_producers(sch, block)
        for producer in producers:
            try:
                sch.compute_inline(producer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumers(
    sch: s_tir.Schedule,
    block: s_tir.schedule.SBlockRV,
):
    while True:
        inlined_cnt = 0
        consumers = _collect_consumers(sch, block)
        for consumer in consumers:
            try:
                sch.compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        for consumer in consumers:
            try:
                sch.reverse_compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumer_chain(
    sch: s_tir.Schedule,
    block: s_tir.schedule.SBlockRV,
):
    auto_inline_consumers(sch, block)
    remaining_consumers = sch.get_consumers(block)

    if len(remaining_consumers) != 0:
        # Some blocks have failed to be inlined to the producer cache-write stage.
        # This could be due to another producer block that has not been scheduled.
        for c in remaining_consumers:
            for p in sch.get_producers(c):
                if sch.get(p) != sch.get(block):
                    auto_inline_producers(sch, p)
                    sch.compute_inline(p)

        # Try inlining into the cache-write stage again, this time it should succeed.
        auto_inline_consumers(sch, block)

    msg = "There are some consumers of the cache-write stage that are not properly inlined."
    assert len(sch.get_consumers(block)) == 0, msg


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


@dataclass
class IterTrait:
    kind: IterKind
    extent: Expr


def _is_one(x: Expr) -> bool:
    return isinstance(x, tvm_tirx.ir.IntImm) and x.value == 1


def make_iter_fusion_index_map(
    traits: list[IterTrait],
    kind_order: list[IterKind],
) -> tirx.IndexMap:
    fused_iters: dict[IterKind, Expr] = {}
    input_iters: list[tirx.Var] = []
    for i, trait in enumerate(traits):
        v_i = tirx.Var(f"i{i}", trait.extent.ty)
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: list[tirx.Expr] = [
        fused_iters.get(kind, tvm_tirx.ir.IntImm(traits[0].extent.ty, 0)) for kind in kind_order
    ]

    return tirx.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: s_tir.SBlock) -> tuple[list[IterTrait]] | None:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : s_tir.SBlock
        The block to be analyzed

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: list[Range]) -> set[Var]:
        axes: set[Var] = set()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes = axes.union(set(undefined_vars(r.begin)))
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: dict[Var, IterTrait] = {}
    for iter_var in block.iter_vars:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.dom.extent):
            kind = IterKind.kIter_T
        elif iter_var.iter_type == iter_var.DataPar:
            if var in A_axes and var in B_axes and var in C_axes:
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.iter_type == s_tir.IterVar.CommReduce:
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.dom.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in A_axes]
    B_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in B_axes]
    C_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in C_axes]
    block_traits = [traits[i.var] for i in block.iter_vars]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(block: s_tir.SBlock) -> tuple[tirx.IndexMap, ...] | None:
    """Get index maps for the block

    Parameters
    ----------
    block : s_tir.SBlock
        The block to be analyzed

    Returns
    -------
    index_maps : Optional[Tuple[tirx.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    A_index_map = make_iter_fusion_index_map(
        A_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K]
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits, [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K]
    )
    C_index_map = make_iter_fusion_index_map(
        C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J]
    )
    matmul_index_map = make_iter_fusion_index_map(
        block_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K]
    )

    return (
        matmul_index_map,
        A_index_map,
        B_index_map,
        C_index_map,
    )


def get_sblock_info(sch: s_tir.Schedule, block: s_tir.schedule.SBlockRV) -> SBlockInfo:
    def _iter_kind(loop: s_tir.IterVar) -> str:
        return {s_tir.IterVar.DataPar: "S", s_tir.IterVar.CommReduce: "R"}.get(loop.iter_type, "O")

    def _is_reduction_block(block: s_tir.schedule.SBlockRV):
        for iter_var in sch.get(block).iter_vars:
            if _iter_kind(iter_var) == "R":
                return True
        return False

    return SBlockInfo(
        name=sch.get(block).name_hint,
        iters=[
            IterInfo(
                kind=_iter_kind(iter_var),
                var=iter_var.var,
                dom=iter_var.dom.extent,
                loop_rv=loop_rv,
            )
            for loop_rv, iter_var in zip(sch.get_loops(block), sch.get(block).iter_vars)
        ],
        block_rv=block,
        reduction_block=_is_reduction_block(block),
    )


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the main computation block
    def is_reduction(block: SBlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: SBlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


class MetalMatmul(GPUScheduleRule):
    """
    The schedule rule for Metal matmul computation.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule | None:
        from tvm.s_tir.tensor_intrin.metal import (  # pylint: disable=import-outside-toplevel
            get_simdgroup_intrin_group,
        )

        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None
        sch = s_tir.Schedule(func)
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        # Step 0. Configs
        block_size_x: int = 16
        block_size_y: int = 16
        block_size_k: int = 32
        micro_size: int = 8
        warp_size: int = 32
        ty_len: int = 1
        tz_len: int = 4
        vector_size: int = 4

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        # Reindex first and than analyze the index map
        main_block = reduction_blocks[0]
        reindex_a = sch.reindex(main_block, ("read", 0))
        reindex_b = sch.reindex(main_block, ("read", 1))
        reindex_c = sch.reindex(main_block, ("write", 0))

        index_maps = get_index_map(sch.get(main_block))
        assert index_maps is not None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        sch.transform_layout(reindex_a, ("write", 0), a_index_map)
        sch.transform_layout(reindex_b, ("write", 0), b_index_map)
        sch.transform_layout(reindex_c, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                ty_len * block_size_x,
                tz_len * block_size_y,
                block_size_k,
            ],
        )

        # Step 3. Schedule matmul to use simdgroup intrinsics
        batch, i, j, k = sch.get_loops(main_block)
        bx, ty, i0, i1 = sch.split(i, [None, ty_len, block_size_x // micro_size, micro_size])
        by, tz, j0, j1 = sch.split(j, [None, tz_len, block_size_y // micro_size, micro_size])
        k0, k1, k2 = sch.split(k, [None, block_size_k // micro_size, micro_size])
        sch.reorder(bx, by, ty, tz, k0, k1, i0, j0, i1, j1, k2)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(batch, "blockIdx.z")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tz, "threadIdx.z")

        def fetch_to_shared(block, idx):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, k0, preserve_unit_loops=True)
            fused = sch.fuse(*sch.get_loops(block_read)[-2:])
            _, _tz, _ty, _tx, vec = sch.split(fused, [None, tz_len, ty_len, warp_size, vector_size])

            sch.bind(_tz, "threadIdx.z")
            sch.bind(_ty, "threadIdx.y")
            sch.bind(_tx, "threadIdx.x")
            sch.vectorize(vec)

            return block_read

        a_g2s = fetch_to_shared(main_block, 0)
        b_g2s = fetch_to_shared(main_block, 1)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to SIMD-group fragments
        A_simdgroup = sch.cache_read(main_block, 0, "metal.simdgroup")
        B_simdgroup = sch.cache_read(main_block, 1, "metal.simdgroup")
        sch.compute_at(A_simdgroup, k1)
        sch.compute_at(B_simdgroup, k1)

        C_simd2s = sch.cache_write(main_block, 0, "metal.simdgroup")
        C_s2g = sch.cache_write(C_simd2s, 0, "shared")
        sch.reverse_compute_at(C_simd2s, tz, preserve_unit_loops=True)
        sch.reverse_compute_at(C_s2g, by, preserve_unit_loops=True)

        intrin_group = get_simdgroup_intrin_group(
            load_scope="shared",
            store_scope="shared",
            dtype="float16",
            trans_a=False,
            trans_b=True,
        )
        sch.transform_layout(B_simdgroup, ("write", 0), lambda s, i, j: (s, j, i))

        def tensorize_block(block: s_tir.schedule.SBlockRV, intrin: str):
            *_, i, j = sch.get_loops(block)
            io, ii = sch.split(i, [None, micro_size])
            jo, ji = sch.split(j, [None, micro_size])
            sch.reorder(io, jo, ii, ji)
            sch.tensorize(ii, intrin)

        C_init = sch.decompose_reduction(main_block, k0)
        tensorize_block(A_simdgroup, intrin_group["load_a"])
        tensorize_block(B_simdgroup, intrin_group["load_b"])
        tensorize_block(C_simd2s, intrin_group["store"])
        tensorize_block(C_init, intrin_group["init"])

        *_, i, j, k = sch.get_loops(main_block)
        sch.tensorize(i, intrin_group["compute"])

        auto_inline_consumer_chain(sch, C_s2g)
        fused = sch.fuse(*sch.get_loops(C_s2g)[-2:])
        _, _tz, _ty, _tx, vec = sch.split(fused, [None, tz_len, ty_len, warp_size, vector_size])
        sch.bind(_tz, "threadIdx.z")
        sch.bind(_ty, "threadIdx.y")
        sch.bind(_tx, "threadIdx.x")
        sch.vectorize(vec)

        return sch


class Matmul(GPUScheduleRule):
    """The schedule rule for matmul-like computation"""

    @dataclass
    class Config:
        block_size_x: int = 8
        block_size_y: int = 8
        vthread_x: int = 1
        vthread_y: int = 1
        micro_size_x: int = 4
        micro_size_y: int = 4
        micro_size_k: int = 8
        vector_size: int = 1
        unroll: int = 256  # 0 means no unroll
        use_shared: bool = True
        storage_align: bool = False
        inner_x: bool = False

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        if target.kind.name == "cuda" or target.kind.name == "rocm":
            return Matmul.Config(
                block_size_x=8,
                block_size_y=16,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=4,
                micro_size_y=4,
                micro_size_k=16,
                vector_size=2,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
        elif target.kind.name == "opencl" and (
            ("android" in str(target.host)) or ("adreno" in str(target.attrs))
        ):
            return Matmul.Config(
                block_size_x=32,
                block_size_y=4,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=8,
                micro_size_y=2,
                micro_size_k=16,
                vector_size=8,
                unroll=16,
                use_shared=False,
                storage_align=False,
                inner_x=True,
            )
        else:
            return Matmul.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule | None:
        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None
        sch = s_tir.Schedule(func)
        config = self.get_configs(target)
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)

        main_block_info = get_sblock_info(sch, main_block)
        iter_infos = main_block_info.iters
        if not get_index_map(block_stmt):
            return None

        # Checks if it's a inner reduction by getting the last matrix's inner Index
        def is_inner_reduction(block_stmt, iter_infos):
            end_it = block_stmt.reads[-1].region[-1].begin
            return {it.var: it.kind for it in iter_infos}.get(end_it, "O") == "R"

        if (
            target.kind.name == "opencl"
            and (("android" in str(target.host)) or ("adreno" in str(target.attrs)))
        ) and not is_inner_reduction(block_stmt, iter_infos):
            ret = self.sch_outer_reduction(sch, config, main_block, blocks)
            if ret is not None:
                return ret

        # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        # Reindex first and than analyze the index map
        reindex_a = sch.reindex(main_block, ("read", 0))
        reindex_b = sch.reindex(main_block, ("read", 1))
        reindex_c = sch.reindex(main_block, ("write", 0))

        index_maps = get_index_map(sch.get(main_block))
        assert index_maps is not None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        sch.transform_layout(reindex_a, ("write", 0), a_index_map)
        sch.transform_layout(reindex_b, ("write", 0), b_index_map)
        sch.transform_layout(reindex_c, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 1. Use the specialized Metal schedule when available.
        if target.kind.name == "metal":
            try:
                return MetalMatmul().apply(func, target, _)
            except:  # pylint: disable=bare-except
                pass

        # Step 2. Schedule matmul
        y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y
        x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x
        if config.inner_x:
            sch.pad_einsum(
                main_block,
                [1, y_kernel_size, x_kernel_size, config.micro_size_k],
            )
            batch, y, x, k = sch.get_loops(main_block)
        else:
            sch.pad_einsum(
                main_block,
                [1, x_kernel_size, y_kernel_size, config.micro_size_k],
            )
            batch, x, y, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(
            y, [None, config.vthread_y, config.block_size_y, config.micro_size_y]
        )
        bx, vx, tx, xi = sch.split(
            x, [None, config.vthread_x, config.block_size_x, config.micro_size_x]
        )
        ko, ki = sch.split(k, factors=[None, config.micro_size_k])
        reordered_loops = [by, bx, vy, vx, ty, tx, ko, ki] + (
            [yi, xi] if config.inner_x else [xi, yi]
        )
        sch.reorder(*reordered_loops)
        by = sch.fuse(batch, by)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        inner_loop = config.micro_size_x if config.inner_x else config.micro_size_y
        if inner_loop % config.vector_size == 0:
            _, v = sch.split(reordered_loops[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.unroll > 0:
            sch.annotate(tx, ann_key="tirx.auto_unroll_max_step", ann_val=config.unroll)
            sch.annotate(tx, ann_key="tirx.unroll_explicit", ann_val=1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        if config.micro_size_x % config.vector_size == 0:
            _, v = sch.split(sch.get_loops(l2g)[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.use_shared:

            def _cooperative_fetch(index, vec_len):
                block = sch.cache_read(main_block, index, "shared")
                num_loops = len(sch.get_loops(block))
                sch.compute_at(block, ko, preserve_unit_loops=True)
                loops = sch.get_loops(block)[-num_loops:]
                ty, tx, _, vec = sch.split(
                    sch.fuse(*loops),
                    factors=[config.block_size_y, config.block_size_x, None, vec_len],
                )
                sch.vectorize(vec)
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                if config.storage_align:
                    sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
                return block

            a_g2s = _cooperative_fetch(0, vec_len=config.vector_size)
            b_g2s = _cooperative_fetch(1, vec_len=config.vector_size)

            auto_inline_producers(sch, a_g2s)
            auto_inline_producers(sch, b_g2s)
        else:
            auto_inline_producers(sch, main_block)

        auto_inline_consumer_chain(sch, l2g)

        sch.decompose_reduction(main_block, ko)
        return sch

    def sch_outer_reduction(
        self,
        sch: s_tir.Schedule,
        config: Config,
        reduction_block: s_tir.schedule.SBlockRV,
        blocks: list[s_tir.schedule.SBlockRV],
    ) -> s_tir.Schedule | None:
        """Get vectorization factor"""

        def get_max_factor(n, factors):
            factors = sorted(factors, reverse=True)
            for factor in factors:
                if n % factor == 0:
                    return factor
            return 1

        reduction_loops = sch.get_loops(reduction_block)
        if not len(reduction_loops) == 4:
            return None

        mb, ms, n, k = reduction_loops
        if not (
            isinstance(
                Range.get_extent_from_begin_end(sch.get(n).begin, sch.get(n).end),
                tvm_tirx.ir.IntImm,
            )
            and isinstance(
                Range.get_extent_from_begin_end(sch.get(mb).begin, sch.get(mb).end),
                tvm_tirx.ir.IntImm,
            )
            and not isinstance(
                Range.get_extent_from_begin_end(sch.get(ms).begin, sch.get(ms).end),
                tvm_tirx.ir.IntImm,
            )
        ):
            return None

        Threads_X, Threads_Y, VecSize, Unroll_M = (
            config.block_size_x,
            config.block_size_y,
            config.vector_size,
            config.unroll,
        )

        VecSize = min(
            get_max_factor(
                Range.get_extent_from_begin_end(sch.get(n).begin, sch.get(n).end) // Threads_X,
                [1, 2, 4, 8],
            ),
            VecSize,
        )
        dequant_block = None
        matmul_block = reduction_block
        epilogue_block = None
        if blocks[-1] is not matmul_block:
            epilogue_block = blocks[-1]
        for blk in blocks[:-1]:
            if "dequantize" in sch.get(blk).name_hint:
                dequant_block = blk
            elif blk is not matmul_block:
                sch.compute_inline(blk)

        block = sch.reindex(reduction_block, ("read", 0))
        sch.pad_einsum(reduction_block, [1, Unroll_M, 1, 1])
        sch.compute_inline(block)
        trans_block, matmul_reindex = (
            sch.get_producers(matmul_block)[0],
            sch.get_consumers(matmul_block)[0],
        )

        if epilogue_block is not None:
            sch.compute_inline(matmul_reindex)
            matmul_reindex = epilogue_block

        sch.transform_layout(
            trans_block,
            ("write", 0),
            T.index_map(lambda i0, i1, i2: (i0, i1 // Unroll_M, i2, i1 % Unroll_M)),
        )

        # transpose block schedules
        # sch.set_scope(trans_block, 0, "global.texture-1d")
        tb, tn, tk = sch.get_loops(trans_block)
        tbx, ttx = sch.split(tk, [None, Threads_X])
        tby, tty, tc = sch.split(tn, [None, Threads_Y, Unroll_M])
        sch.bind(tb, "blockIdx.z")
        sch.bind(tby, "blockIdx.y")
        sch.bind(tbx, "blockIdx.x")
        sch.bind(tty, "threadIdx.y")
        sch.bind(ttx, "threadIdx.x")
        sch.reorder(tb, tby, tbx, tty, ttx, tc)
        sch.vectorize(tc)

        mb, ms, n, k = sch.get_loops(matmul_block)
        m = sch.fuse(mb, ms)
        bx, tx, vec = sch.split(n, [None, Threads_X, VecSize])
        by, ty, unr = sch.split(m, [None, Threads_Y, Unroll_M])
        k1, k2, k3 = sch.split(k, [None, 4, 8])
        sch.reorder(bx, by, tx, ty, k1, k2, k3, unr, vec)
        sch.set_scope(matmul_block, 0, "local")
        if dequant_block is not None:
            sch.compute_at(dequant_block, k3)
            sch.set_scope(dequant_block, 0, "local")
        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

        inp = sch.cache_read(matmul_block, read_tensor_index=0, storage_scope="local")
        sch.compute_at(inp, k3, preserve_unit_loops=True)
        sch.vectorize(sch.get_loops(inp)[-1])

        sch.unroll(unr)
        sch.unroll(k3)

        if dequant_block is not None:
            Aq_local = sch.cache_read(dequant_block, read_tensor_index=0, storage_scope="local")
            sch.compute_at(Aq_local, k2, preserve_unit_loops=True)
            sch.vectorize(sch.get_loops(Aq_local)[-1])
            As_local = sch.cache_read(dequant_block, read_tensor_index=1, storage_scope="local")
            sch.compute_at(As_local, k1, preserve_unit_loops=True)
            sch.vectorize(sch.get_loops(As_local)[-1])
            sch.vectorize(sch.get_loops(dequant_block)[-1])

        sch.reverse_compute_at(matmul_reindex, ty)
        o_ur, o_vec = sch.get_loops(matmul_reindex)[-2:]
        sch.vectorize(o_vec)
        sch.unroll(o_ur)
        sch.decompose_reduction(matmul_block, k1)

        return sch
