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
# ruff: noqa: RUF005
"""layout_transform scheduling rule for CUDA."""

import math
from collections import deque

import tvm_ffi

import tvm_tirx
from tvm.s_tir import meta_schedule
from tvm.s_tir.schedule import ExprRV, LoopRV, SBlockRV, Schedule

## Tiling layout transforms:
# Assume we have an input shape of [A, B, C, D] and want to layout transform
# ABCD --> DBAC so the output shape would be [D, B, A, C].
#
# Consider reading from the input tensor in a cache-friendly fashion on CPU. We would
# expect a loop structure like:
# lAr, lBr, lCr, lDr = T.grid(A, B, C, D)
#
# Meanwhile consider writing to the output tensor in a cache-friendly fashion on CPU:
# lDw, lBw, lAw, lCw = T.grid(D, B, A, C)
#
# Clearly in many scenarios it is impossible to guarantee contiguous writes and reads
# within a single loop due to non-adjacent dimensions. Instead we work on transposing some
# small sub-tensor of our input writing and then reading from shared memory. We must now
# construct our submatrix so that reading and writing can both be done with some contiguous
# access in global memory.
#
# Consider the case of a 2D transpose. For example [1024, 2048] -> [2048, 1024].
# We note that if we deal with a submatrix of shape [32, 32] which corresponds
# to the dimension of our input tensor, then rows of the submatrix are contiguous
# in the input tensor. Meanwhile, columns of our submatrix are contiguous in our
# output vector. Therefore, with this tile shape we have opportunity to read
# contiguously in our input tensor and write to shared memory, and write contiguously
# to our output tensor.
#
# The multiple dimensional case has a similar analogue. We want to allocate shared
# memory per block of [`tile_size`, `tile_size`]. We want the inner most dimension
# of our shared memory to correspond to contiguous reads from the input tensor and
# the outer dimension to correspond to contiguous writes into the output tensor.
#
# In terms of the loop structure reading from the input tensor, the inner most loops
# of our tile must correspond to the inner most dimensions of the input shape,
# while the outer dimensions correspond to the inner most dimensions of the output shape.
# To obtain an inner tile with this loop structure we factor out a contiguous `tile_size`
# chunk of our loop in the shape of interest.
#
# An example is probably best to show this idea:
# Let's say we want a layout transform of ABCD --> DCAB. With shape
# [1024_a, 2_b, 32_c, 8_d] --> [8_d, 32_c, 1024_a, 2_b]
#
# And tile size 32.
#
# Then we initially have a coalesced-read loop pattern of:
# T.grid(1024_a, 2_b, 32_c, 8_d)
#
# To obtain an inner tile of 32, we factor 4 from 32_c and 8 from 8_d:
# T.grid(1024_a, 2_b, 8_c1, 1_d1, 4_c2t, 8_d2t)
# T.grid(1024_a, 2_b, 8_cr, 1_dr, 32_dim1)
#
# To obtain an outer tile of 32, we factor from B then A to follow contiguous write
# pattern:
#
# T.grid(64_a1, 1_b1, 8_cr, 1_dr, 16_a2t, 2_b2t, 32_dim1)
# T.grid(64_ar, 1_br, 8_cr, 1_dr, 32_dim0, 32_dim1)
#
# Which allows us to read a tile with our wanted properties.
# For writing we use the existing analysis infrastructure to generate the structure for writing.


def tile_layout_transform(
    sch: Schedule,
    block_read: SBlockRV,
    block_write: SBlockRV,
    src_layout: str,
    dst_layout: str,
    input_shape: list[int],
    tile_size: ExprRV,
) -> tuple[SBlockRV, SBlockRV]:
    """
    High level tiling for layout transform block. Mutates sch in place.

    Parameters
    ----------
    sch:
        The initial schedule. We expect `block_read` and `block_write` to correspond to
        the blocks which reads and writes from global memory respectively. We also expect
        block_read's initial loops to follow

    block_read:
        The block which reads from global memory and writes to shared memory tensor.

    block_write:
        The block which writes to global memory and reads from shared memory tensor.

    src_layout :
        The src_layout, each character should appear once and also appear in dst_layout.
        There should be not numeric characters and refer to potentially implicit reshapes.
        E.g. the transform NCHW --> NCHW4c really implies NCcHW --> NCHWc. In this case
        src_layout should be NCcHW.

    dst_layout:
        The dst_layout. There should not be numeric characters, e.g. NCHW4c becomes NCHWc.

    input_shape:
        The input shape after applying potentially implicit reshapes. Should match the loop
        extants corresponding to src_layout.

    tile_size:
        The tile size of read and writes. There will be tile_size threads per block, each of which
        reads up to tile_size elements.

    Returns
    -------
    ret:
        A tuple of the block that writes to global memory, and the block that reads from
        global memory.
    """

    def pad_dimension_to_at_least_number(loop: LoopRV, requested_size: int):
        """E.g. if loop has extant of 8 but we want 10, returns size 10 loop with padding."""
        left, right = sch.split(loop, [None, requested_size])
        return sch.fuse(left, right)

    def pad_dimension_to_factor_of_tile_size(
        loop: LoopRV, initial_size: int, tile_size: int = tile_size
    ) -> tuple[LoopRV, int]:
        """
        Pads loop of given size until it is divisible into tile_size.
        If the given size of the loop is greater than tile size. Do not pad.

        examples:
        - loop_size = 5 , tile_size = 32. loop_size --> 8
        - loop_size = 5 , tile_size = 36. loop_size --> 6
        - loop_size = 8 , tile_size = 32. loop_size --> 8  : since 8 already divides 32.
        - loop_size = 33, tile_size = 32. loop_size --> 33 : since 33 > 32.

        Returns padded loopRV and the new size.
        """
        if tile_size % initial_size == 0:
            return loop, int(initial_size)

        if initial_size > tile_size or initial_size == tile_size:
            return loop, int(initial_size)

        # if initial_size > tile_size return without change, factor = 1
        size = initial_size
        while (tile_size % size) % tile_size > 0:
            size += 1

        return pad_dimension_to_at_least_number(loop, size), int(size)

    def spin_out_factor(
        loops: list[LoopRV], loop_extants: list[int], index: int, factor_needed: int
    ) -> tuple[list[LoopRV], list[int], int]:
        """
        Factor out the requested loop's dimensions to reach the requested factor and
        places the requested factor as the innermost loop.

        Updates the schedule in-place.

        E.g. say we want to factors which eventually multiply to 32 (factor_needed).

        Say we have the index we chose is a loop with an extant of 8.
        E.g. loops / loop_extants = [3, 32, 6, 8], factor_needed = 32, index=3 (dim=8)
            - 8 divides into 32 so we just split up the loop into two loops with extants 1 and 8.
            - we then keep the 1-loop in place and move the new 8-loop to back of the list of loops
            - ending loops / loop_extants = [3, 32, 6, 1, 8], remaining_factor_needed = 32 / 8 = 4

        E.g. loops / loop_extants = [3, 32, 6, 8], factor_needed=32, index=0 (dim=3)
            - 3 does not divide 32, so we pad until the extant divides 32, e.g. 4
            - we then split up the loop into extants 1 and 4, moving the 4 to the back
            - ending loops / loop_extants = [1, 32, 6, 8, 4], remaining_factor_needed = 32 / 4 = 8

        E.g. loops / loop_extants = [3, 32, 6, 8], factor_needed=5, index=3 (dim=8)
            - 8 is larger than 5 so we immediately do the splitting routine.
            - the 8 extant loop becomes loops with extants 2 and 5
            - ending loops / loop_extants = [1, 32, 6, 2, 5], remaining_factor_needed = 5 / 5 = 1

        After updating loop ordering in place, returns the new list of loops, extants, and the
        remaining factor needed.
        """
        cur_loop = loops[index]
        cur_extant = loop_extants[index]

        # Pad loops to divide evenly for factors needed, and split
        new_loop, new_size = pad_dimension_to_factor_of_tile_size(
            cur_loop, cur_extant, tile_size=factor_needed
        )

        split_factor = min(new_size, factor_needed)
        new_loop_split, factored_loop = sch.split(new_loop, [None, split_factor])
        factor_needed = factor_needed // split_factor

        # update caching
        loops[index] = new_loop_split
        loops.append(factored_loop)

        loop_extants[index] = math.ceil(int(new_size) / int(split_factor))
        loop_extants.append(split_factor)

        sch.reorder(*loops)
        return loops, loop_extants, factor_needed

    def factor_dim_in_order(
        indices: list[int],
        loops: list[LoopRV],
        cur_loop_extants: list[int],
        work_needed_inner_loop: int = tile_size,
    ) -> tuple[list[LoopRV], list[int]]:
        """Factors out the loops in the order of indices until we reach needed work.

        Adds new loop factors to the back in reverse order of access. Returns new list
        of loops and their extants.
        """
        for i in indices:
            loops, cur_loop_extants, work_needed_inner_loop = spin_out_factor(
                loops, cur_loop_extants, i, work_needed_inner_loop
            )
            if work_needed_inner_loop == 1:
                break
        return loops, cur_loop_extants

    def get_high_level_loop_structure(
        block_read: SBlockRV, input_shape: list[int], src_layout: str, dst_layout: str
    ):
        """Runs the factorization described above."""
        # index 0 ... rank - 1 will always correspond to original loops
        # perhaps after they have been factored.
        rank = len(input_shape)
        loops = sch.get_loops(block_read)
        cur_loop_extants = list(input_shape)

        # Factor dim0 tile size and fuse things together
        loops, cur_loop_extants = factor_dim_in_order(
            list(range(rank - 1, -1, -1)),
            loops,
            cur_loop_extants,
            work_needed_inner_loop=tile_size,
        )
        # The factors which multiply to tile_size are now in back of our
        # list of loops. However because we added them by traversing the inner
        # dimensions, they are actually reversed order to guarantee the best access
        # so reorder before fusing.
        loops = loops[:rank] + loops[rank:][::-1]
        cur_loop_extants = cur_loop_extants[:rank] + cur_loop_extants[rank::-1]
        sch.reorder(*loops)
        dim0_loop_tiled = sch.fuse(*loops[rank:])
        loops = loops[:rank]
        loops.append(dim0_loop_tiled)
        cur_loop_extants = cur_loop_extants[:rank]
        cur_loop_extants.append(tile_size)

        # Same thing with dim1
        # [:rank + 1], since we placed dim0_loop_tiled in the end which we want to keep
        loops, cur_loop_extants = factor_dim_in_order(
            list(
                src_layout.index(dst_layout[loop_index_dst])
                for loop_index_dst in range(rank - 1, -1, -1)
            ),
            loops,
            cur_loop_extants,
            work_needed_inner_loop=tile_size,
        )
        loops = loops[: rank + 1] + loops[rank + 1 :][::-1]
        cur_loop_extants = cur_loop_extants[: rank + 1] + cur_loop_extants[rank + 1 :: -1]
        sch.reorder(*loops)
        dim1_loop_tiled = sch.fuse(*loops[rank + 1 :])
        loops = loops[: rank + 1]
        loops.append(dim1_loop_tiled)
        cur_loop_extants = cur_loop_extants[: rank + 1]
        cur_loop_extants.append(tile_size)

    # After this we have loops: [loop1, loop2, loop3 ... dim0_tiled, dim1_tiled]
    get_high_level_loop_structure(block_read, input_shape, src_layout, dst_layout)

    # If there are insufficient elements, than dim1_tiled or dim0_tiled might be too small
    # In all likelihood you should use a smaller tile, but I don't want things to crash.
    loops = sch.get_loops(block_read)
    loops[-1] = pad_dimension_to_at_least_number(loops[-1], tile_size)
    loops[-2] = pad_dimension_to_at_least_number(loops[-2], tile_size)

    # We want the dim0 and dim1 parent loops to be the inner most. Right now dim1 is inner-msot
    # and we just need to move dim0 in (last dimension of dst).
    # Recall right now structure is at least [l1 l2 ... ln, dim0_tiled, dim1_tiled]
    # where n >= 2.
    dim0_loop_index = src_layout.index(dst_layout[-1])
    dim0_loop = loops.pop(dim0_loop_index)
    loops = loops[:-3] + [dim0_loop, loops[-3]] + loops[-2:]
    sch.reorder(*loops)

    # After this loops are: [outer_loop (block binding), dim0_tiled, dim1_tiled]
    outer_loop = sch.fuse(*loops[:-2])

    # Now that we have the high level loop structure, we can use reverse_compute_at magic
    # To get the proper loop structure for writing! This is also as coalesced as possible
    # already.
    sch.reverse_compute_at(block_write, outer_loop)

    # Fuse all inner loops for the write into 2 loops, grab inner loops for both read
    # and write block which have locality (we will bind these to threadIdx)
    fused_write_loop = sch.fuse(*sch.get_loops(block_write)[1:])
    _, inner_write_loop = sch.split(fused_write_loop, [None, tile_size])
    inner_read_loop = sch.get_loops(block_read)[-2]

    sch.bind(loop=outer_loop, thread_axis="blockIdx.x")
    sch.bind(loop=inner_write_loop, thread_axis="threadIdx.x")
    sch.bind(loop=inner_read_loop, thread_axis="threadIdx.x")

    return block_write, block_read


def create_cached_read(
    sch: Schedule,
    block_write: SBlockRV,
    orig_input_shape: list[int],
    orig_src_layout: str,
    orig_dst_layout: str,
) -> tuple[SBlockRV, list[int], str, str]:
    """
    Creates the cached read block with expected structure.

    Loop extants should follow the input shape closely. E.g. if the input is [2, 6, 8], we
    expect our loop structure to be T.grid(2, 6, 8). Possibly reshape to handle implicit reshapes,
    in which case we will match the implicit reshape shape.

    Layout transform allows semantics like NCHW --> NCHW4c. Which involves splitting the original C
    axis into contiguous 4-element chunks. This axis is then moved to the end (NCHWc). This is
    guaranteed by the operator to be done without additional padding. To handle this we just split
    the associating axis (prev. type checking ensures C is divisible by 4)in src_layout found in
    block_read. E.g. NCHW -> NCHW4c now becomes NC4cHW -> NCHW4c.

    Note: NCHW4c --> NCHW is not allowed, so the only numeric digits will be in dst.

    The returned layout strings will be santized and made compatible. E.g. NCHW --> NCHW4c becomes
    NCcHW --> NCHWc.

    TODO(AndrewZhaoLuo): Investigate using proper memory alignment to avoid bank conflict.

    Parameters
    ----------
    sch:
        The initial schedule. We expect `block_read`. We also expect
        block_read's initial loops to follow the original input shape.

    block_read:
        The block which reads from global memory and writes to shared memory tensor.

    orig_input_shape:
        The input shape of the input tensor to the function.

    orig_src_layout:
        The original src_layout string.

    orig_dst_layout:
        The original dst_layout string.

    Returns
    -------
    ret:
        A tuple of the cached read block, new input shape of shared memory tensor,
        the new src_layout, and new dst_layout string.
    """
    # Figure out split dimensions, entries are (loop index in src_layout, split amount)
    split_dimensions: list[tuple[int, int]] = []

    # This is without numeric digits, e.g. NCHW4c -> NCHWc
    new_dst_layout = []

    # Use state machine to parse NCHW4c string
    split_size = 0
    for char in orig_dst_layout:
        if char.isnumeric():
            split_size = split_size * 10 + int(char)
        else:
            if char.islower():
                # hit axis like 'c', need to find parent axis 'C' in src_layout
                src_layout_index = orig_src_layout.index(char.upper())
                split_dimensions.append((src_layout_index, split_size))
            split_size = 0
            new_dst_layout.append(char)

    # If no splits were detected we are done
    if len(split_dimensions) == 0:
        block_read = sch.cache_read(block_write, 0, "shared")
        return block_read, orig_input_shape, orig_src_layout, orig_dst_layout

    # Calculate final input shapes, each of these are a single element for unsplit dims
    # and tuples for split dims associated with the two new axis
    input_shape: list[int | tuple] = list(orig_input_shape)
    new_src_layout: list[str | tuple] = list(orig_src_layout)
    for src_layout_split_index, split_factor in split_dimensions:
        dimension_name = orig_src_layout[src_layout_split_index]
        new_src_layout[src_layout_split_index] = (dimension_name, dimension_name.lower())
        input_shape[src_layout_split_index] = (
            orig_input_shape[src_layout_split_index] // split_factor,
            split_factor,
        )

    # Unpack any tuples introduced via appending
    def unpack_list(target_list) -> list:
        output: list = []
        for ele in target_list:
            if isinstance(ele, tuple):
                output.extend(ele)
            else:
                output.append(ele)
        return output

    new_src_layout_str = "".join(unpack_list(new_src_layout))
    new_dst_layout_str = "".join(unpack_list(new_dst_layout))

    # Write block loop extants match
    dst_to_src_map = [new_dst_layout_str.index(dim) for dim in new_src_layout_str]
    block_read = sch.reindex_cache_read(
        block_write,
        read_tensor_index=0,
        index_map=tvm_tirx.tirx.IndexMap.from_func(
            lambda *loops: [loops[dst_to_src_map[i]] for i, _ in enumerate(loops)],
            ndim=len(new_src_layout_str),
        ),
        storage_scope="shared",
    )

    loops_read = sch.get_loops(block_read)
    sch.reorder(
        *[loops_read[new_dst_layout_str.index(dst_dim_name)] for dst_dim_name in new_src_layout_str]
    )
    return block_read, unpack_list(input_shape), new_src_layout_str, new_dst_layout_str


def auto_inline_into(sch: Schedule, start_block: SBlockRV) -> SBlockRV:
    """
    Inlines given start_block's consumers and future dependencies into start_block.

    Parameters
    ----------
    sch:
        The initial schedule.

    start_block:
        The block to inline into, should be a block which reads and writes to global memory, doing
        layout transform.

    Returns
    -------
    ret:
        The new block inlined into it's consumers.
    """
    # Rules defined by DefaultCUDA schedule_rule set.
    autoinline_rule = meta_schedule.schedule_rule.AutoInline(
        into_producer=True,
        into_consumer=False,
        inline_const_tensor=True,
        disallow_if_then_else=False,
        require_injective=False,
        require_ordered=False,
    )

    fringe = deque(sch.get_consumers(start_block))
    visited = set()
    while len(fringe) > 0:
        cur_block = fringe.popleft()
        if cur_block in visited:
            continue

        visited.add(cur_block)
        consumer_blocks = sch.get_consumers(cur_block)
        fringe.extend(consumer_blocks)

        sch = autoinline_rule.apply(sch, cur_block)[0]


def get_max_tile_size(target=None) -> int:
    """Returns the max tile size.

    This is assuming only threads in a warp can have coalesced accesses. 32 is the default if
    no target information can be gotten.
    """
    max_tile_size = 32
    cur_target = target
    if cur_target is not None and "thread_warp_size" in cur_target.attrs:
        max_tile_size = int(cur_target.attrs["thread_warp_size"])
    return max_tile_size


@tvm_ffi.register_global_func("s_tir.meta_schedule.cuda.layout_transform")
def cuda_layout_transform_schedule_rule(
    sch: Schedule, block: SBlockRV, testing_tile_sizes: list[int] | None = None
) -> list[Schedule]:
    """
    Applies tiling scheme to layout transform task (potentially fused with other injective funcs).

    Returned schedules will be the default schedule, as well as tiled versions with tile_size in
    the range of 2,3...threads_per_warp.

    This is assuming only threads in a warp can have coalesced accesses. 32 is the default if
    no target information can be gotten.

    Parameters
    ----------
    sch:
        The initial schedule.

    block:
        The block corresponding to the layout transform.
        Should be a block which reads and writes to global memory, doing layout transform.

    testing_tile_sizes:
        A list of tile sizes to try, overriding normal settings. For testing. None means
        ignore. Else overrides normal settings of tile sizes to try.

    Returns
    -------
    ret:
        A list of new schedules to try.
    """
    # Info needed for tiling
    src_layout = sch.get_sref(block).stmt.annotations["src_layout"]
    dst_layout = sch.get_sref(block).stmt.annotations["dst_layout"]
    input_shape = [int(c) for c in sch.get_sref(block).stmt.annotations["input_shape"]]

    schedules = []

    # Always include the default schedules which will be handled via AutoBind schedule rule
    # Except during testing
    if not testing_tile_sizes:
        schedules.append(sch)

    sch = sch.copy()

    # Inline consumers of the layout transform into the layout transform block.
    # Normally default for injective schedules but must manually be called in new schedule rule
    # for consumers of the layout transform. TODO(AndrewZhaoLuo): Figure out why this is the case.
    auto_inline_into(sch, block)

    # Setup up basic structure of schedule of creating read into shared mem, before applying tiling
    # Outer loop structure of read block matches that of src_layout
    # E.g. if input_shape is [4, 6, 8]. Loops for read block will be
    # for i, j, k in T.grid(4, 6, 8):
    #     ...
    # Read block will read from global memory coalesced at the start
    # Assume write to output global memory is coalesced in block_write
    #
    # This also handles the case where there is an implicit reshape going on.
    # e.g. NCHW -> NCHW4c which is equivalent to reshaping NCHW
    # to NCcHW and then applying the new layout where the extant of c is 4.
    # Grab final input shape and src and dst layouts with possible implicit reshape.
    block_read, input_shape, src_layout, dst_layout = create_cached_read(
        sch, block, input_shape, src_layout, dst_layout
    )

    # Try tile size 2,3...threads_per_warp as tile size of 1 has no coaslescing.
    if testing_tile_sizes is None:
        tile_sizes = list(
            range(2, get_max_tile_size(sch.mod[sch.func_working_on].attrs.get("target")) + 1)
        )
    else:
        tile_sizes = testing_tile_sizes

    for tile_size in tile_sizes:
        new_sch = sch.copy()
        tile_layout_transform(
            new_sch, block_read, block, src_layout, dst_layout, input_shape, tile_size
        )
        schedules.append(new_sch)

    return schedules
