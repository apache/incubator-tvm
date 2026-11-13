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

tcgen05.ld / tcgen05.st
=======================

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``registers, tmem for ld; tmem, registers for st; split forms use two
TMEM operands``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.
   * - ``imm_half_splitoff``
     - static int; required only for 16x32bx2
     - Native address immediate from the first TMEM half-access to the second; validated by the selected PTX form.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
-------------------

Register operands are b32 words. Ordinary TMEM operands are 32-bit cells; pack/unpack
forms expose the documented extra halfword axis with 16-bit elements. The shape and xN
determine counts; storage dtype does not select packing.

Operand specification
---------------------

These warp instructions transfer a fixed image between TMEM and registers. The opcode
specifies the row/bit shape, repetition qualifier ``.xN``, and optional pack/unpack
mode. The tile repeats that exact instruction over caller-supplied views. A load takes
``registers,tmem``; a store takes ``tmem,registers``. Both require all 32 lanes of each
participating warp.

Native images
~~~~~~~~~~~~~~

For the ordinary b32 forms both operands end in ``(row,column)``. Let ``l`` be the lane
ID and ``r`` a register index within one instruction:

.. list-table::
   :header-rows: 1

   * - Shape
     - Atom extents for xN
     - Registers per lane
     - Register r contains coordinate
   * - ``32x32b``
     - ``(32,N)``
     - N
     - ``(l,r)``
   * - ``16x64b``
     - ``(16,2N)``
     - N
     - ``(l//4+8*(l%2), l//2%2+2*r)``
   * - ``16x128b``
     - ``(16,4N)``
     - 2N
     - ``(l//4+8*(r%2), l%4+4*(r//2))``
   * - ``16x256b``
     - ``(16,8N)``
     - 4N
     - ``(l//4+8*(r//2%2), r%2+2*(l%4)+8*(r//4))``

TMEM rows and columns are consecutive physical lanes/cells inside each atom. Each warp
can access its own 32-lane slab; a 16-row instruction may select either half explicitly.
Warp scope therefore requires a statically selected warp. Warpgroup/CTA scope exposes
the warp as a group axis. Register slots may be separated as long as their owners match
this table.

Legality conditions
-------------------

Regions have the specified native images and matching group/repeat prefixes. Register
ownership follows the per-shape lane/register mapping. TMEM rows align to the atom row
count and belong to the executing warp slab. Packed origins select a cell's low half.
Split regions share a backing allocation and their addresses differ by
imm_half_splitoff. No pred keyword is accepted.

Code generation
---------------

For each repeat, enumerate the selected shape's native register coordinates, bind those
slots, and encode the warp-relative TMEM address. Emit one selected tcgen05.ld/st,
adding the split immediate where applicable. Thread synchronization, tcgen05.wait and
producer/consumer fences are explicit caller operations.

Examples
--------

Example: Layout B without changing its physical mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layout B represents logical ``(64,128)`` values across 128 physical lanes and 64
columns. Logical ``[i,j]`` is at lane ``i+64*(j//64)``, column ``j%64``. The following
round trip isolates transfers from the MMA producer: it uses one CTA and writes/reads
this layout with explicit st/ld instructions.

.. code-block:: python

    from tvm.script import tirx as T
    from tvm.backend.cuda.axis import tid_in_wg
    from tvm.backend.cuda.layout import tmem_datapath_layout
    from tvm.tirx.layout import TileLayout, S

    layout_b = tmem_datapath_layout("B", 64, 128)
    register_layout = TileLayout(S[(64, 2, 64) : (1 @ tid_in_wg, 64 @ tid_in_wg, 1)])


    @T.function
    def tmem_roundtrip(A: T.Tensor((128, 64), "float32"), Out: T.Tensor((128, 64), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        address = T.alloc_shared((1,), "uint32", layout="default")
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(64)
            )
        T.cuda.cta_sync()
        mem = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((64, 128), "float32", scope="tmem", layout=layout_b)]
        )
        src = T.alloc_tensor(ty_args=[
            T.Tensor((64, 128), "float32", scope="local", layout=register_layout)
        ])
        dst = T.alloc_tensor(ty_args=[
            T.Tensor((64, 128), "float32", scope="local", layout=register_layout)
        ])
        local_src = src.view(64, layout="default")
        for j in range(64):
            local_src[j] = A[tid, j]
        mv = mem.rearrange("(wh r) (wl c) -> (wl wh) r c", wh=2, r=32, wl=2)
        sv = src.rearrange("(wh r) (wl c) -> (wl wh) r c", wh=2, r=32, wl=2)
        dv = dst.rearrange("(wh r) (wl c) -> (wl wh) r c", wh=2, r=32, wl=2)
        T.cuda.tile.apply(T.ptx["tcgen05.st.sync.aligned.32x32b.x64.b32"], mv, sv, scope="warpgroup")
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cta_sync()
        T.cuda.tile.apply(T.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"], dv, mv, scope="warpgroup")
        T.ptx.tcgen05.wait__ld.sync.aligned()
        local_dst = dst.view(64, layout="default")
        for j in range(64):
            Out[tid, j] = local_dst[j]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(64))

The views have shape ``(4,32,64)`` with ``warp=2*wl+wh``. There are no repeat axes: each
of the four warps issues one x64 store and one x64 load. No physical rearrangement or
fragment copy is inserted. The explicit register layout maps logical ``[i,j]`` to thread
``i+64*(j//64)`` and local slot ``j%64``. The physical views expose each thread's 64
contiguous registers.

.. code-block:: python

    import numpy as np
    import tvm

    x = np.arange(8192, dtype="float32").reshape(128, 64)
    compiled = tvm.tirx.compile(tmem_roundtrip, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    compiled(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x)

Pack/unpack is a different TMEM image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tcgen05.ld...pack::16b.b32`` and ``tcgen05.st...unpack::16b.b32`` keep the ordinary
b32 register image. The TMEM operand instead has suffix ``(row,column,half)`` with
``half=2`` and a 16-bit storage type. The two halves of register word ``[row,column]``
correspond to the **low 16 bits of two different TMEM cells**, at columns ``2*column``
and ``2*column+1``.

In 16-bit layout units, the column and half strides are respectively ``4 @ TCol`` and
``2 @ TCol``. Using strides 2 and 1 would instead select both halves of one dense cell
and would not satisfy the instruction. Hardware performs the pack/unpack; the tile emits
no software conversion. The upper 16 bits left by an unpack are not part of its defined
16-bit payload.

Split accesses: 16x32bx2
~~~~~~~~~~~~~~~~~~~~~~~~

This form uses register suffix ``(access,row,column)=(2,16,N)``; lane ``l`` owns
``(l//16,l%16,r)``. Supply two separate TMEM regions and the explicit native immediate
``imm_half_splitoff``. Each TMEM suffix is ``(16,N)`` or ``(16,N,2)`` for packing. For
example, for an already prepared warpgroup view ``mem`` with shape ``(4,32,16)``::

    T.cuda.tile.apply(
        T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x4.b32"],
        registers, mem[:, :16, :4], mem[:, 16:32, 8:12],
        imm_half_splitoff=(16 << 16) + 8, scope="warpgroup",
    )

Both halves share the same TMEM backing allocation. The second encoded address equals
the first plus the immediate; no extra sixteen-row shift is implied by lane 16. For
stores, operand order is ``mem_low,mem_high,registers``. One instruction handles both
accesses for each repeat.

Completion and producer dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``tcgen05.wait::ld`` before consuming loaded registers and ``tcgen05.wait::st``
before relying on stores. For an MMA/cp producer, complete its commit/mbarrier protocol
and required thread-sync fence before issuing a TMEM load; ``wait::ld`` does not
complete a preceding MMA. Allocation, deallocation, fences and synchronization remain
explicit.

Implementation: ``python/tvm/backend/cuda/tile/patterns/tmem.py``.
