..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

cp.async.bulk.tensor / cp.reduce.async.bulk.tensor
==================================================

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``shared, global_region_or_tensormap for load;
global_region_or_tensormap, shared for store/reduction``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.
   * - ``box``
     - static tuple[int]; required
     - One extent per opcode rank, outer-to-inner, each 1..256.
   * - ``coords``
     - tuple of runtime integers; required
     - Starting coordinates relative to the descriptor view, outer-to-inner.
   * - ``coord_steps``
     - static nested integer tuples; default ()
     - One rank-length increment vector for each group/repeat axis; must be supplied when these axes exist.
   * - ``format``
     - static dtype str; default operand dtype
     - Descriptor representation; may be tf32 over float32 storage.
   * - ``element_strides``
     - static tuple[int]; default all ones
     - Per-dimension sampling steps, each 1..8.
   * - ``swizzle``
     - static int; default 0
     - 0/1/2/3 means none/32B/64B/128B; must match the shared image and an external descriptor.
   * - ``interleave``
     - static int; default 0
     - 0/1/2 means none/16B/32B; changes the canonical image.
   * - ``l2_promotion``
     - static int; default 0
     - Descriptor L2 promotion mode, 0..3.
   * - ``oob_fill``
     - static int; default 0
     - 0: zero; 1: NaN for supported floating-point formats.
   * - ``gather4``
     - four runtime integers; required for gather4/scatter4
     - The four explicit row coordinates.
   * - ``mbar``
     - runtime pointer; required for loads
     - Completion barrier pointer in the opcode's shared address space.
   * - ``cta_mask``
     - runtime integer; required for multicast
     - Native multicast mask, uint16/uint32 according to the opcode.
   * - ``cache_policy``
     - runtime uint64; required with cache hint
     - Native cache-policy operand.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
-------------------

The descriptor format and interleave mode define the storage representation.
Non-interleaved shared suffix is ceil(box/element_strides); gather4/scatter4 uses
(4,box[1]). Interleaved and padded FP4 images are specified below.

Operand specification
---------------------

TMA transfers a descriptor-defined global box to or from a shared-memory image. Select
direction, rank, tile/gather/scatter mode, multicast, cache qualifiers, and any
reduction operator in the opcode. ``apply`` encodes the specified descriptor and issues
that instruction at explicit coordinates; it does not plan a decomposition of an
arbitrary tensor layout.

Operands and descriptor fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a load the operands are ``shared, global``; for a store/reduction they are ``global,
shared``. The global operand is either a rectangular global tensor region or an ordinary
pointer to a caller-encoded TensorMap. In the latter case the caller is responsible for
matching its fields to the tile contract; lowering does not inspect or select opaque
descriptors.

Without interleave, the innermost step is one.
   * - ``interleave``
     - 0/1/2: none/16B/32B. The global and shared images change as described below.
   * - ``l2_promotion``, ``oob_fill``
     - Explicit descriptor policies; both default to zero.
   * - ``mbar`` / ``cta_mask`` / ``cache_policy``
     - Runtime operands required by load / multicast / cache-hint forms, respectively.

Without interleave, the global region has exactly the opcode rank, unit innermost
stride, and suitably aligned outer byte strides. Its dimensions and outer strides can be
runtime values and are passed to the host descriptor encoder. Known invalid fields are
rejected rather than silently promoted, folded, transposed or reinterpreted as another
dtype.

The shared suffix is the box image: ``ceil(box[i]/element_strides[i])`` per dimension,
compact before swizzling. Every atom's unswizzled origin is 128-byte aligned, even for
swizzle zero; a swizzled backing base is aligned to the whole swizzle period. Swizzled
subregions use their inverse-swizzled origins so the hardware applies the phase once.

Legality conditions
-------------------

Global tensor rank matches opcode rank, plus the explicit channel-slice axis for
interleave. Shared group/repeat extents are static and its image matches the descriptor.
Global dimensions/strides may be runtime values checked by the host encoder. Every
unswizzled shared origin is 128B aligned; swizzled allocations align to their period. An
opaque TensorMap cannot be inspected: its declared fields are the caller's contract.

Code generation
---------------

For a global tensor region, create the host TensorMap from its dimensions/strides and
explicit fields. For each group/repeat, add coord_steps to coords, reverse coordinate
order for PTX, undo the shared origin swizzle once, and emit one selected native
instruction. gather4/scatter4 substitutes the explicit row coordinates. Host descriptor
setup is separate from device repeat loops; commit/wait, barrier accounting and election
remain caller code.

Examples
--------

Example: two explicit 8x16 boxes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def tma_load(A: T.Tensor((16, 16), "uint32"), Out: T.Tensor((16, 16), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((2, 8, 16), "uint32", layout="default", align=128)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(1024))
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"],
                shared,
                A,
                box=(8, 16),
                coords=(0, 0),
                coord_steps=((8, 0),),
                mbar=mbar.ptr_to([0]),
            )
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        for g, r in T.grid(2, 8):
            if lane < 16:
                Out[g * 8 + r, lane] = shared[g, r, lane]

The shared shape is ``(repeat=2,box_row=8,box_col=16)``. One elected lane issues two
instructions at global coordinates ``(0,0)`` and ``(8,0)``. Each copies 512 bytes; one
phase of the barrier accounts for 1024 bytes. Descriptor construction is host
initialization associated with this call. It is not repeated on the device for each box.

.. code-block:: python

    import numpy as np
    import tvm

    x = np.arange(256, dtype="uint32").reshape(16, 16)
    compiled = tvm.tirx.compile(tma_load, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    compiled(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x)

Stores, reductions, and completion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The corresponding store spelling is
``cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group``. Reduction uses, for
example, ``cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group``. The
coordinate/box contract is the same, with global destination first. Complete shared
initialization and issue ``T.ptx.fence.proxy.async_.shared__cta()`` before the transfer;
the issuing thread then calls ``T.ptx.cp.async_.bulk.commit_group()`` and
``T.ptx.cp.async_.bulk.wait_group(0)``. Initialize a reduction destination to the
intended starting value. No mbarrier operand is used for these stores.

Loads use their supplied mbarrier, with explicit initialization, expected transaction
bytes, and phase wait. Multicast adds the explicit CTA mask and the corresponding
cluster/barrier setup. The tile call elects no thread and inserts neither waits nor
descriptor prefetch. A desired prefetch is an ordinary ``T.ptx.prefetch.tensormap``
call.

Gather4 and scatter4
~~~~~~~~~~~~~~~~~~~~

These forms use a 2D descriptor with ``box=(1,width)`` and a shared image ``(4,width)``.
The four row coordinates are ordinary runtime operands in ``gather4``; they need not be
consecutive. For a prepared ``shared`` of shape ``(4,32)`` and a matching barrier, an
issue can be written as::

    T.cuda.tile.apply(
        T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"],
        shared, A, box=(1,32), coords=(0,0),
        gather4=(row,row+3,row+7,row+11), mbar=mbar.ptr_to([0]),
    )

For uint32 this is one 512-byte gather4 issue. Scatter4 uses global destination first
and the matching store opcode/completion protocol. Gather/scatter coordinate selection
is caller logic.

Interleave and packed FP4
~~~~~~~~~~~~~~~~~~~~~~~~~

With ``interleave=1`` or ``2``, a rank-3..5 opcode takes global axes
``(N,C_slices,*spatial,Ci)``. ``Ci`` holds one 16B/32B slice and is implicit in the
descriptor rank. Its elements are contiguous and the innermost spatial stride is one
slice. Interleave 2 requires swizzle 1 and 32-byte global alignment. One issue selects
**one** C slice, even if a larger C box field is supplied; channel iteration requires an
explicit repeat and C step.

For example, uint16/interleave-2 with ``box=(1,1,8)`` has shared atom ``(1,1,8,16)``.
Add a leading repeat of three and ``coord_steps=((0,1,0),)`` to issue three successive
channel slices. Gather4/scatter4 do not use interleave. On B200, 16B interleave with
swizzle 1/2/3 can trap, including when issued through raw PTX.

``format="float4_e2m1fn"`` selects the load-only ``16U4_ALIGN16B`` format. The inner
global dimension is a multiple of 128, the inner box is 128, and global origins/outer
strides are 32-byte aligned. Each sixteen nibbles occupy the low eight bytes of a
sixteen-byte shared block. The remaining eight bytes are undefined padding; barrier
accounting counts payload bytes, not this padding. Swizzle 0/3 and gather4 use this
explicit packed image.

Implementation: ``python/tvm/backend/cuda/tile/patterns/tma.py``.
