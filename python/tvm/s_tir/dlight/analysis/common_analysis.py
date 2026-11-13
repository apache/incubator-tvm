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

# pylint: disable=missing-function-docstring, missing-class-docstring
# pylint: disable=unused-argument, unused-variable
"""Analysis on TIR blocks, loops and functions."""

import logging
from collections import namedtuple
from typing import Literal

import tvm_ffi
from tvm_ffi import get_global_func

import tvm_tirx
from tvm import s_tir
from tvm.s_tir import Schedule, match_shape_stride_pattern
from tvm.s_tir.schedule import SBlockRV
from tvm_tirx import ir, tirx
from tvm_tirx.ir import Range, TensorRegion
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.target.target import Target

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class IterInfo:
    """Information about a loop/iter var."""

    kind: Literal["S", "R", "O"]
    var: tirx.Var
    _dom: tirx.Expr
    loop_rv: s_tir.schedule.LoopRV

    def __init__(
        self,
        kind: Literal["S", "R", "O"],
        var: tirx.Var,
        dom: tirx.Expr,
        loop_rv: s_tir.schedule.LoopRV,
    ):
        """Construct an IterInfo object."""
        self.kind = kind
        self.var = var
        self._dom = dom
        self.loop_rv = loop_rv

    @property
    def dom(self) -> int | tirx.Expr:
        """The iteration domain of the loop."""
        return int(self._dom) if isinstance(self._dom, tvm_tirx.ir.IntImm) else self._dom

    def __str__(self) -> str:
        return f'Iter("{self.kind}", {self.dom})'

    def __repr__(self) -> str:
        return str(self)


get_sblockrealize = get_global_func("s_tir.schedule.GetSBlockRealize")
# TensorIndex Types
Index = namedtuple("Index", ["sub"])  # c
RemIndex = namedtuple("RemIndex", ["sub", "div"])  # c%len
DivIndex = namedtuple("DivIndex", ["sub", "div"])  # c//len
MergeIndex = namedtuple("MulIndex", ["dom", "mul", "sub"])  # co*len + cb
TensorIndex = list[Index | RemIndex | DivIndex | MergeIndex | None]


class TensorInfo:
    "Information about Tensor. Provides useful analysis"

    tensor_region: TensorRegion
    shape: tuple[int]
    assoc_lps: list[s_tir.schedule.LoopRV | None]
    assoc_lps_info: list[tvm_tirx.ir.For | None]

    def __init__(
        self,
        sch: s_tir.Schedule,
        block_rv: s_tir.schedule.SBlockRV,
        tensor_region: TensorRegion,
        lps: list[s_tir.schedule.LoopRV] | None,
    ):
        block = sch.get(block_rv)
        if lps is None:
            lps = sch.get_loops(block_rv)
        loops = [sch.get(lp) for lp in lps]
        iter_vars = [Var.var for Var in block.iter_vars]
        iter_values = get_sblockrealize(sch, block_rv).iter_values
        lpvar_lp = dict([loop.var, lp] for loop, lp in zip(loops, lps))
        var_lp = dict(zip(iter_vars, [lpvar_lp.get(val, None) for val in iter_values]))

        def extract_index_types(tensor: TensorRegion) -> TensorIndex:
            tensor_index = []
            for expr in tensor.region:
                expr = expr.begin
                dim = None
                if isinstance(expr, tirx.expr.Add) and ir.is_prim_var(expr.b):
                    var_add = expr.b
                    if (
                        isinstance(expr, tirx.expr.Mul)
                        and ir.is_prim_var(expr.a)
                        and isinstance(expr.b, tvm_tirx.ir.IntImm)
                    ):
                        mul = expr.b
                        var_mul = expr.a
                        dim = MergeIndex(var_mul, mul, var_add)
                elif (
                    isinstance(expr, tirx.expr.FloorMod)
                    and ir.is_prim_var(expr.a)
                    and isinstance(expr.b, tvm_tirx.ir.IntImm)
                ):
                    dim = RemIndex(expr.a, expr.b)
                elif (
                    isinstance(expr, tirx.expr.FloorDiv)
                    and ir.is_prim_var(expr.a)
                    and isinstance(expr.b, tvm_tirx.ir.IntImm)
                ):
                    dim = DivIndex(expr.a, expr.b)
                elif ir.is_prim_var(expr):
                    dim = Index(expr)
                tensor_index.append(dim)
            return tensor_index

        indexes = extract_index_types(tensor_region)
        assoc_lps = [
            (
                var_lp.get(getattr(idx, "sub"), None)
                if not isinstance(idx, DivIndex) and idx is not None
                else None
            )
            for idx in indexes
        ]

        self.tensor_region = tensor_region
        self.assoc_lps = assoc_lps
        self.assoc_lps_info = [(sch.get(lp) if lp is not None else None) for lp in assoc_lps]
        self.shape = tensor_region.source.shape

    def get_scope(self) -> MemoryScope:
        return self.tensor_region.source.scope()

    def get_vecsize(self, tensor_index: int = 0, vbits: int = 128):
        if self.assoc_lps_info[-1] is None:
            return None

        vlp_extent = int(
            Range.get_extent_from_begin_end(
                self.assoc_lps_info[-1].begin, self.assoc_lps_info[-1].end
            )
        ) & ~(
            int(
                Range.get_extent_from_begin_end(
                    self.assoc_lps_info[-1].begin, self.assoc_lps_info[-1].end
                )
            )
            - 1
        )
        vtensor_extent = int(self.shape[-1]) & ~(int(self.shape[-1]) - 1)

        return min(vlp_extent, vtensor_extent, vbits // self.tensor_region.source.dtype.bits)

    def __str__(self) -> str:
        return f"TensorInfo({self.tensor_region})"

    def __repr__(self) -> str:
        return str(self)


class SBlockInfo:
    """Information about a TIR block."""

    name: str
    iters: list[IterInfo]
    block_rv: s_tir.schedule.SBlockRV
    _reduction_block: bool

    def __init__(
        self,
        name: str,
        iters: list[IterInfo],
        block_rv: s_tir.schedule.SBlockRV,
        reduction_block: bool = False,
    ):
        """Construct a SBlockInfo object."""
        self.name = name
        self.block_rv = block_rv
        self.iters = iters
        self._reduction_block = reduction_block

    def dom(self) -> list[int | tirx.Expr]:
        """The iteration domain of the block."""
        return [i.dom for i in self.iters]

    def read_tensors(self, sch: s_tir.Schedule) -> list[TensorInfo]:
        block_stmt = sch.get(self.block_rv)
        lps = sch.get_loops(self.block_rv)
        return [TensorInfo(sch, self.block_rv, tensor, lps) for tensor in block_stmt.reads]

    def write_tensors(self, sch: s_tir.Schedule) -> list[TensorInfo]:
        block_stmt = sch.get(self.block_rv)
        lps = sch.get_loops(self.block_rv)
        return [TensorInfo(sch, self.block_rv, tensor, lps) for tensor in block_stmt.writes]

    def dom_kind(self) -> str:
        """The iteration domain kind of the block, for example, SSSS, SSSR."""
        return "".join(i.kind for i in self.iters)

    def is_injective(self) -> bool:
        """Whether the SBlock is injective, i.e. all its iteration domains are injective."""
        return all(k == "S" for k in self.dom_kind())

    def is_elementwise(self, sch: s_tir.Schedule) -> bool:
        """Whether the SBlock is elementwise, i.e. trivial mapping between read/write region"""

        def _check_unit_var_range(dom: ir.Range, var: tirx.Var) -> bool:
            return dom.begin.same_as(var) and dom.extent == 1

        if not self.is_injective():
            return False
        block = sch.get(self.block_rv)
        if len(block.reads) != 1 or len(block.writes) != 1:
            return False
        r_region = block.reads[0].region
        w_region = block.writes[0].region
        if len(r_region) != len(w_region):
            return False
        for var, r_dom, w_dom in zip(block.iter_vars, r_region, w_region):
            if not _check_unit_var_range(r_dom, var) or not _check_unit_var_range(w_dom, var):
                return False
        return True

    def get_loops(self) -> list[s_tir.schedule.LoopRV]:
        return [iter_info.loop_rv for iter_info in self.iters]

    def is_reduction(self) -> bool:
        """Whether the SBlock is a reduction workload."""
        # TODO(@junrushao): distinguish GEMV and reduction
        return self._reduction_block

    def is_layout_transform(self, sch: s_tir.Schedule) -> bool:
        """Whether the SBlock can be considered having a Layout Transform Pattern"""
        return (
            all(k == "S" for k in self.dom_kind())
            and len(self.write_tensors(sch)) == 1
            and len(self.read_tensors(sch)) == 1
            and not self.is_elementwise(sch)
            and not get_global_func("s_tir.schedule.HasIfThenElse")(sch.get(self.block_rv))
        )

    def is_data_pad(self, sch: s_tir.Schedule) -> bool:
        """Whether the SBlock can be considered having a data pad pattern"""
        return (
            all(k == "S" for k in self.dom_kind())
            and len(self.write_tensors(sch)) == 1
            and len(self.read_tensors(sch)) == 1
            and not self.is_elementwise(sch)
            and len(self.write_tensors(sch)[0].tensor_region.region)
            == len(self.read_tensors(sch)[0].tensor_region.region)
            and get_global_func("s_tir.schedule.HasIfThenElse")(sch.get(self.block_rv))
        )

    def is_convolution(self) -> bool:
        """Whether a SBlock can be considered having Convolution Pattern"""
        raise NotImplementedError

    def is_pool(self) -> bool:
        """Whether a SBlock can be considered having Pooling Pattern"""
        raise NotImplementedError

    def is_gemv(self) -> bool:
        """Whether the SBlock is a GEMV workload."""
        raise NotImplementedError

    def is_gemm(self) -> bool:
        """Whether the SBlock is a GEMM workload."""
        raise NotImplementedError

    def __str__(self) -> str:
        return f'SBlockInfo("{self.name}", "{self.dom_kind()}", {self.dom()})'

    def __repr__(self) -> str:
        return str(self)


_normalize_function = get_global_func("s_tir.schedule.NormalizeFunction")


def normalize_function(sch: s_tir.Schedule) -> list[SBlockInfo] | None:
    """Normalize the function to normal form"""
    try:
        result = _normalize_function(sch)
        if result is None:
            return None
    except Exception:  # pylint: disable=broad-except
        return None

    def _iter_kind(i: s_tir.IterVar) -> str:
        return {
            s_tir.IterVar.DataPar: "S",
            s_tir.IterVar.CommReduce: "R",
        }.get(i.iter_type, "O")

    blocks: list[SBlockInfo] = []
    for block, loops, iters, is_reduction in zip(*result):
        blocks.append(
            SBlockInfo(
                name=sch.get(block).name_hint,
                iters=[
                    IterInfo(
                        kind=_iter_kind(iter),  # type: ignore
                        var=iter.var,
                        dom=iter.dom.extent,
                        loop_rv=loop,
                    )
                    for loop, iter in zip(loops, iters)
                ],
                block_rv=block,
                reduction_block=is_reduction,
            )
        )
    return blocks


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


def _assert_gpu_target(target: Target):
    if "gpu" not in target.keys:
        raise ValueError(f"Expect a GPU target, but got {target}")


def get_max_threads_per_block(target: Target) -> int:
    _assert_gpu_target(target)
    max_threads_per_block = None
    for name in ["max_threads_per_block", "max_num_threads"]:
        if max_threads_per_block is None:
            max_threads_per_block = target.attrs.get(name, None)
    if max_threads_per_block is None:
        max_threads_per_block = 64
    return int(max_threads_per_block)


TARGET_KIND_TO_DEFAULT_MAX_SMEM = {
    "cuda": 49152,
    "rocm": 65536,
    "metal": 32768,
    "opencl": 16384,
    "vulkan": 16384,
}


def get_max_shared_memory_per_block(target: Target) -> int:
    _assert_gpu_target(target)
    max_shared_memory_per_block = target.attrs.get("max_shared_memory_per_block", None)
    if max_shared_memory_per_block is not None:
        return int(max_shared_memory_per_block)

    # Layered fallback strategy for targets that do not carry this attribute
    # 1) Use explicit target attrs provided (handled above).
    # 2) Fall back to backend defaults matching target-kind defaults/tag defaults.
    # 3) Use a conservative GPU default as last resort.
    default_smem = TARGET_KIND_TO_DEFAULT_MAX_SMEM.get(target.kind.name, 16384)
    logger.warning(
        "Target %s missing 'max_shared_memory_per_block'; using %d bytes.",
        target.kind.name,
        default_smem,
    )
    return int(default_smem)


def get_root_block(sch: Schedule, func_name: str = "main") -> SBlockRV:
    body = sch.mod[func_name].body
    if body is None or len(body) != 1 or not isinstance(body[0], s_tir.SBlockRealize):
        raise ValueError(
            f"The function body is expected to be the root block, but got:\n"
            f"{sch.mod[func_name].body}"
        )
    return sch.get_sblock(body[0].block.name_hint)


def collect_block_iter_vars_used_in_access_region(
    block: s_tir.SBlock, region: list[ir.Range]
) -> set[tirx.Var]:
    """Collect the block iter variables used in the access region of a tensor region."""
    tir_vars = set()
    for expr in region:
        assert expr.extent == 1
        tir_vars |= collect_vars_used_in_prim_expr(expr.begin)
    tir_vars &= set(iter_var.var for iter_var in block.iter_vars)
    return tir_vars


def collect_vars_used_in_prim_expr(expr: tirx.Expr) -> set[tirx.Var]:
    """Collect the variables used in the Expr."""
    tir_vars = set()

    def _collect_tir_var(expr: tirx.Var):
        tir_vars.add(expr)

    tvm_ffi.structural_walk(expr, (tirx.Var, _collect_tir_var), order="post")
    return tir_vars


def detect_dominant_read(block: s_tir.SBlock) -> tirx.Expr:
    """Detect the dominant read indices in the block."""
    dominant_read = None
    num_read_iters = -1
    for tensor_region in block.reads:
        tir_vars = collect_block_iter_vars_used_in_access_region(block, tensor_region.region)
        if num_read_iters < len(tir_vars):
            num_read_iters = len(tir_vars)
            dominant_read = tensor_region
    assert dominant_read is not None
    return _tensor_region_offset(dominant_read)


def _tensor_region_offset(region):
    """Linear element offset of a region's first element."""
    tensor = region.source
    pattern = match_shape_stride_pattern(tensor.ty)
    assert pattern is not None
    offset = pattern.elem_offset
    for axis, bounds in enumerate(region.region):
        offset += bounds.begin * pattern.strides[axis]
    return tvm_tirx.sym.Analyzer().simplify(offset)


def is_broadcast_epilogue(
    sch: s_tir.Schedule,
    block: s_tir.schedule.SBlockRV,
    epilogue: s_tir.schedule.SBlockRV,
) -> bool:
    """Check if the epilogue block is a broadcast pattern"""
    write_tensors = {r.source for r in sch.get(block).writes}
    epilogue_iters = {i.var: i for i in sch.get(epilogue).iter_vars if i.dom != 1}
    for tensor_region in sch.get(epilogue).reads:
        if tensor_region.source not in write_tensors:
            continue
        tir_vars = collect_block_iter_vars_used_in_access_region(
            sch.get(epilogue), tensor_region.region
        )
        if len(tir_vars) < len(epilogue_iters):
            return True
    return False
