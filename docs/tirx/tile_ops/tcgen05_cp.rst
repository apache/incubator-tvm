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

tcgen05.cp
==========

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_tmem, src_shared``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
-------------------

Both regions use scalar 32-bit storage words. Opcode shape gives (rows,words); multicast
determines destination replicas. Decompression uses the same b32 footprint but
explicitly changes the packed payload semantics.

Operand specification
---------------------

``tcgen05.cp`` copies a descriptor-defined shared image into TMEM. It is issued by a
single thread, not collectively by the warps whose TMEM lanes receive the data. Select
shape, CTA group, multicast and optional decompression directly in the opcode.

TMEM shape and replicas
~~~~~~~~~~~~~~~~~~~~~~~~

Operands are ``dst_tmem, src_shared``, both with suffix ``(row,b32_word)`` and 32-bit
storage elements. An earlier axis is an explicit repeat.

.. list-table::
   :header-rows: 1

   * - Shape / multicast
     - Atom extents
     - TMEM lanes
   * - ``128x256b``
     - ``(128,8)``
     - Consecutive 128 lanes
   * - ``128x128b``
     - ``(128,4)``
     - Consecutive 128 lanes
   * - ``4x256b``
     - ``(4,8)``
     - 0, 32, 64, 96 relative to the origin
   * - ``64x128b.warpx2::02_13``
     - ``(64,4)``
     - 64 consecutive lanes, replicated 64 lanes apart
   * - ``64x128b.warpx2::01_23``
     - ``(64,4)``
     - Logical row r maps to ``64*(r//32)+r%32``, replicated 32 lanes apart
   * - ``32x128b.warpx4``
     - ``(32,4)``
     - 32 consecutive lanes, four replicas 32 lanes apart

The destination layout must state these replicas explicitly; the pattern does not infer
multicast. Columns are consecutive b32 cells with a 128-bit aligned origin. The whole
footprint, including replicas, must stay inside the 128-lane, 512-column TMEM address
space. A CTA-group-2 opcode makes both CTAs copy from their own corresponding shared
storage into their own TMEM. The launch, allocation and commit must use the matching
CTA-group protocol.

Shared descriptor address walk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared layout determines internal byte strides LDO/SDO and swizzle mode 0/1/2/3.
These are not call parameters. Before swizzling, row ``r`` starts at ``(r//8)*SDO +
(r%8)*(16 << swizzle)`` bytes relative to the descriptor origin. Without swizzling, word
``w`` adds ``(w//4)*LDO + (w%4)*4``. With swizzling, words are contiguous and add
``4*w``; hardware ignores LDO in that case.

Descriptor strides are nonnegative 16-byte multiples below 2^18; origins are 16-byte
aligned. Swizzled bases require 256/512/1024-byte period alignment for modes 1/2/3. The
complete physical walk is checked, including the origin's swizzle phase. A 4-row logical
region still needs a backing storage block large enough for the descriptor's eight-row
organization.

Legality conditions
-------------------

Both regions end in the selected (rows,words). Shared must match one of the descriptor
address walks in the operand specification. TMEM row/column mapping and replicas match the opcode exactly.
Column origin is 128-bit aligned and the full image fits 128 lanes and 512 columns.
Shared origins are 16B aligned, descriptor strides are nonnegative 16B multiples below
2^18, and swizzled allocations align to their full period.

Code generation
---------------

Extract LDO/SDO/swizzle from the shared region using
:ref:`cuda-matrix-descriptor-extraction`. For each repeat, encode the shared atom
origin, compute the TMEM address from TLane/TCol, and emit one selected tcgen05.cp. A
single elected thread issues it; allocation, proxy fencing, commit, wait and
deallocation remain explicit caller code.

Examples
--------

Example: one 128x128b copy and readback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tvm.script import tirx as T
    from tvm.backend.cuda.axis import TLane, TCol
    from tvm.tirx.layout import S, TileLayout

    tmem_layout = TileLayout(S[(128, 4) : (1 @ TLane, 1 @ TCol)])


    @T.function
    def copy_to_tmem(A: T.Tensor((128, 4), "uint32"), Out: T.Tensor((128, 4), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        shared = T.alloc_shared((128, 4), "uint32", layout="default", align=16)
        address = T.alloc_shared((1,), "uint32", layout="default")
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(32)
            )
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        for j in range(4):
            shared[tid, j] = A[tid, j]
        T.cuda.cta_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        mem = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((128, 4), "uint32", scope="tmem", layout=tmem_layout)]
        )
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.cp.cta_group::1.128x128b"],
                mem,
                shared,
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        word = T.alloc_local((1,), "uint32", layout="default")
        for j in range(4):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(j))
            T.ptx.tcgen05.wait__ld.sync.aligned()
            Out[tid, j] = word[0]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(32))

One cp instruction copies 2048 bytes. Its commit triggers one barrier arrival, rather
than a TMA-style expected-byte transaction. The scalar x1 readback issues are caller
code and independently expose each TMEM word.

.. code-block:: python

    import numpy as np
    import tvm

    x = np.arange(512, dtype="uint32").reshape(128, 4)
    compiled = tvm.tirx.compile(copy_to_tmem, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    compiled(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x)

Native FP4/FP6 decompression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Append ``.b8x16.b4x16_p64`` or ``.b8x16.b6x16_p32`` to the opcode. The b32 footprint
above stays unchanged: each 16-byte shared block contains sixteen packed values followed
by eight bytes (FP4) or four bytes (FP6) of padding. The destination contains sixteen
expanded bytes. Four-bit values gain two low zero bits; six-bit values gain two high
zero bits. A 256-bit row contains two such blocks. The pattern emits one native
decompression instruction per repeat, with no software unpacking loop.

For instance, the example's opcode can be explicitly replaced with
``tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64`` **when the source data has been
prepared in that packed format**. The reference then compares the expanded values, not
the original words or padding. A dtype name alone neither prepares the payload nor
chooses this qualifier.

Allocation, shared proxy fencing, election, commit, wait and deallocation are caller
code. ``pred`` gates the cp but does not adjust a surrounding completion protocol.
Implementation: ``patterns/tmem_copy.py`` and ``patterns/descriptors.py`` under
``python/tvm/backend/cuda/tile``.
