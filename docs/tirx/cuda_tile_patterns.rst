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

Explicit CUDA tile patterns
================================================================================

Choose an instruction, prepare zero-copy operand views, then apply it::

    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], registers, memory,
                      scope="thread")

The instruction and modifiers determine the atom contract. The caller supplies regions
with the canonical ``group × repeat × atom`` coordinates described below. Lowering
checks the contract and emits one specified PTX instruction per repeat coordinate, in
row-major repeat order. Lowering unrolls static repeats. It does not infer a different
opcode, width, matrix count, transpose mode, or coordinate permutation.

CUDA exposes two tile operations: ``apply`` for an explicitly selected PTX instruction
and ``ewise`` for an explicitly supplied scalar TIRx function. The raw ``T.ptx``
namespace remains available for scalar/address operands. There is no automatic
instruction selection or fallback to the retired CUDA dispatch algorithms. Trainium's
API and lowering are unchanged.

This page defines common conventions; each instruction guide is its complete contract
reference. The :doc:`tile_ops` instruction guides provide complete kernels, caller
views, synchronization, and numerical checks, organized by PTX instruction family.

Group, repeat, and atom axes
--------------------------------------------------------------------------------

A thread instruction at ``scope="thread"`` has no group prefix. At ``scope="warp"`` it
has a leading lane group axis; at ``scope="warpgroup"`` its groups are ``(warp_in_group,
lane)``; at ``scope="cta"`` they are ``(warp_in_cta, lane)``. Extents come from the
active execution scope, including its explicitly guarded subrange. Warp-collective
instructions consume the lane axis inside their atom instead: their group prefix is
empty at warp scope, one warp axis at warpgroup/CTA scope. All 32 lanes must be active.
Cluster scope requires an explicit launch cluster and prepends its CTA axes in z/y/x
order, omitting undeclared leading axes. A one-dimensional cluster therefore has
``(cta_x, warp, lane)`` groups for thread instructions. Local tensor ownership uses
``cbx/cby/cbz`` for these CTA coordinates.

For example, 64 threads, three 128b loads per thread::

    # A and R both have shape (2, 32, 3, 4).
    registers = TileLayout(S[(2,32,3,4):(1 @ warpid,1 @ laneid,8,2)])
    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R, A, scope="cta")

The first two coordinates bind hardware IDs, the third generates three unrolled calls
per thread, and the last selects four words of one instruction. The separated register
slots are legal; the four memory words must be aligned and contiguous. A changed opcode
or modifier is always written at the call site.

Views such as ``sub``, ``permute``, ``view(dtype)``, and ``rearrange`` prepare these
coordinates without moving data. A failed contract requires a different explicit view,
caller-written preparation, or a different explicit instruction. It never invokes a
permutation/factorization solver.

Units, regions and generated code
---------------------------------

Layout storage strides are in elements of the declared tensor dtype. A b32 container
counts as one element even when it packs two b16 values. Descriptor strides are in
bytes; every formula labels its unit. Symbols in a layout pattern describe values read
from the layout, not additional keyword arguments.

A tensor argument denotes its full region. For a region with start vector o, physical
coordinates are layout.apply(o + group/repeat/atom coordinates), plus the tensor element
offset where applicable. Extents are positive static integers; starts may be symbolic
when bounds and alignment can be proved. Group/repeat extents agree across operands,
except for documented MMA M/N/K sharing. A default effective layout follows TensorType's
ordinary rules; the tile API does not create a different physical image to satisfy an
instruction.

One repeat means one core native instruction for apply. Address preparation may include
descriptor arithmetic or mapa, as stated in the instruction guide. Counts describe tile
lowering, before later compiler optimizations. ewise instead emits one scalar function
evaluation per element. Registers named by layout offsets are storage slots; machine
register allocation is not specified.

Every instruction guide lists all public instruction parameters, with defaults and
conditional requirements. Generic IR construction keywords (attrs, ty, ty_args, loc) are
not hardware parameters. Unknown instruction keywords are rejected. There is no
universal pred support: consult the selected family.

128-bit global load
--------------------------------------------------------------------------------

``ld.global.v4.b32``, ``scope="thread"``:

* Both regions have shape ``(*repeat, 4)`` and scalar 32-bit elements.
* The source is global memory. For each repeat, its four words have addresses
  ``base + 0, base + 1, base + 2, base + 3`` (in words); the base is 16B aligned.
* The destination is four distinct registers of the executing thread. Their
  storage offsets may be separated or permuted. No thread axes are accepted.
* The operation copies raw bits; it performs no numeric conversion.

The earlier example keeps its original float32 tensor layout::

    original = TileLayout(S[(15, 2, 7, 2):(4, 1, 60, 2)])
    A_std = A.rearrange("i b j d -> j i (d b)")
    R_std = R.rearrange("i b j d -> j i (d b)")
    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R_std, A_std,
                      scope="thread")

``A_std`` has shape ``(7,15,4)`` and addresses ``60*j + 4*i + word``. The repeat axes
are ``j`` and ``i``: this emits 7*15 = 105 vector loads. The destination may instead use
original strides ``(8,2,120,4)``; each instruction then writes slots separated by two,
without a remap copy.

ldmatrix with interleaved rows and separated registers
--------------------------------------------------------------------------------

``ldmatrix.sync.aligned.m8n8.x2.shared.b16``, ``scope="warp"``:

* Both regions have shape ``(*repeat, 2, 8, 4)``. The suffix is
  ``(matrix, row, word)``.
* Each word is a 32-bit container holding two adjacent 16-bit matrix values.
  The pattern accepts float32/int32/uint32 containers as raw bits.
* In shared memory, each ``(matrix,row)`` is four consecutive words with a
  16B-aligned start. Different rows can be separated; they need not form a
  contiguous 8x8 allocation.
* Destination element ``(matrix,row,word)`` belongs to lane ``4*row+word``.
  Each lane receives two words; the register slots are selected by its layout.
* All 32 lanes must execute the instruction. Shared initialization and any
  preceding synchronization belong to the caller.

Use the original shared shape ``(16,4,2)`` and strides ``(8,1,128)``::

    S_std = SM.rearrange("(r m) w k -> k m r w", r=8, m=2)
    registers = TileLayout(S[(2,2,8,4):(8,2,4 @ laneid,1 @ laneid)])
    # R is declared over 12 caller-allocated words per thread, using registers.
    T.cuda.tile.apply(T.ptx["ldmatrix.sync.aligned.m8n8.x2.shared.b16"],
                      R, S_std, scope="warp")

``S_std[k,m,r,w] = SM[2*r+m,w,k]``. Its strides are ``(128,8,16,1)``: first matrix =
even original rows; second matrix = odd original rows. For each ``k``, lanes 0..7
provide the eight matrix-0 row addresses, lanes 8..15 provide the eight matrix-1 row
addresses, and lanes 16..31 duplicate those valid addresses. These address providers are
distinct from result owners.

For lane ``l``, the complete output mapping is::

    R_raw[0]  = SM[2*(l//4),   l%4, 0]
    R_raw[2]  = SM[2*(l//4)+1, l%4, 0]
    R_raw[8]  = SM[2*(l//4),   l%4, 1]
    R_raw[10] = SM[2*(l//4)+1, l%4, 1]

There are two repeat coordinates, ``k=0,1``, so this is exactly two ``.x2``
instructions. Slots 1,3,4,5,6,7,9,11 remain untouched. Lowering binds a physical view of
the caller's registers and passes these slots directly as PTX output operands; it
allocates no intermediate tile.

Memory and register instructions
--------------------------------

For the complete call parameters, operand specification and generated code, see
:doc:`tile_ops/ld_st` and the arithmetic/cvt/mov pages in :doc:`tile_ops`.

Scalar compositions
--------------------------------------------------------------------------------

Define exactly the scalar expression to apply::

    @T.function(private=True)
    def silu_affine(x: T.float32, scale: T.float32, bias: T.float32) -> T.float32:
        y = x / (T.float32(1) + T.exp(-x))
        return y * scale + bias

    T.cuda.tile.ewise(silu_affine, R, R, T.float32(2), T.float32(1), scope="cta")

The atom is one scalar element. Tensor operands share the group/repeat shape; scalar
inputs broadcast. A tensor broadcast is an explicit zero-stride view. Types must match
the function parameters and result. The scalar function may contain scalar bindings and
returns, including conditional returns. Its typed expression is serialized as
``LambdaExpr`` and inlined at each register store. No shared/global staging,
synchronization, or packed-vector selection occurs. Shared-memory compositions
explicitly load into registers, run ``ewise``, and store back, with caller-owned
synchronization.

Asynchronous copy
-----------------

For the complete call parameters, operand specification and generated code, see
:doc:`tile_ops/cp_async` and :doc:`tile_ops/cp_async_bulk`.

TMA
---

For the complete call parameters, operand specification and generated code, see
:doc:`tile_ops/cp_async_bulk_tensor`.

TMEM transfers
--------------

See :doc:`tile_ops/tcgen05_ld_st` and :doc:`tile_ops/tcgen05_cp` for the full contracts.

``.16x32bx2`` has register suffix ``(access,row,column)=(2,16,N)``. Lane ``l`` owns
``(l//16,l%16,r)`` for register ``r``. Its two TMEM access regions are explicit
operands, each with suffix ``(16,N)`` (or ``(16,N,2)`` for packing)::

    T.cuda.tile.apply(
        T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x4.b32"],
        registers, memory[:, :16, :4], memory[:, 16:32, 8:12],
        imm_half_splitoff=(16 << 16) + 8, scope="warpgroup",
    )

For a store the operand order is ``memory_low, memory_high, registers``. Both regions
share the same backing TMEM tensor. The second region's encoded base address must equal
the first plus ``imm_half_splitoff``. There is no implicit sixteen-row offset for the
second access. The immediate can select different rows, columns, or both, within the
executing warp's TMEM partition. Each repeat emits exactly one instruction, including
both half-accesses. Wait, fence, allocation and deallocation instructions remain at the
call site.

.. _cuda-matrix-descriptor-extraction:

Shared matrix descriptors
--------------------------------------------------------------------------------

``tcgen05.cp`` and ``tcgen05.mma`` obtain shared descriptors from their canonical
operand views. LDO/SDO are internal byte strides; swizzle is an internal hardware
encoding. None is an additional tile call parameter. The opcode and MMA major order
remain explicit user choices.

Extraction first tries the standard swizzle represented by ComposeLayout, then the
remaining legal modes in ascending order. cp permits modes 0..3; MMA permits 0..3 except
that 32-bit MN-major requires mode 4. For each mode, inverse-swizzle the bit addresses
and obtain row-group and column-chunk strides from fixed boundary coordinates. The
entire address walk must match, including every group/repeat and the region's nonzero
origin. This finite encoding check does not permute logical coordinates or select
another opcode.

An LDO unused by the atom is encoded as 16 bytes; an unused SDO is zero. Extraction
never probes outside the requested atom to determine those fields. A small region may
not expose all swizzle bits; physical equivalence over the complete access footprint is
sufficient. Prefer the layout's stated mode when valid; otherwise choose the first fully
matching legal mode. Symbolic origins remain address expressions; descriptor strides
must be compile-time constants.

See :doc:`tile_ops/tcgen05_cp` and :doc:`tile_ops/tcgen05_mma` for address formulas,
field roles and alignment. The descriptor encodes the unswizzled origin so hardware
applies its phase exactly once. Base-offset and absolute-LDO modes are outside the
current tile contract.

Warp MMA
--------

For the complete call parameters, operand specification and generated code, see
:doc:`tile_ops/mma_sync`.

tcgen05 MMA
-----------

For the complete call parameters, operand specification and generated code, see
:doc:`tile_ops/tcgen05_mma`.

Choosing instructions
--------------------------------------------------------------------------------

Choose each instruction and its canonical view at the call site:

.. list-table:: Instruction choices
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Instruction or composition
   * - ld/st, register copy
     - ``apply`` with a written ld/st width or mov opcode.
   * - ldmatrix/stmatrix
     - ``apply`` with matrix count and transpose qualifier in the opcode.
   * - Scalar/packed arithmetic and casts
     - ``apply`` with the exact arithmetic/cvt form and native register representation.
   * - exp, silu, scale/bias combinations
     - ``ewise`` with the caller's scalar TIRx function and explicit constants.
   * - Arithmetic on shared memory
     - Explicit shared load, register ``apply``/``ewise``, then shared store.
   * - cp.async, DSMEM, TMA
     - ``apply`` with explicit chunk/box/coordinates and runtime completion operands.
   * - tcgen05.ld/st/cp
     - ``apply`` with explicit shape, repeat count and multicast qualifiers.
   * - mma/tcgen05.mma
     - ``apply`` with explicit matrix mode, canonical M/N/K repeats and operand layouts.

Keep barriers, thread election, packing, scale-factor preparation and pipeline
scheduling in caller code. Changing a view changes only logical coordinates; when the
desired physical arrangement requires a copy, write that copy explicitly. Unsupported
hardware modes remain available through scalar ``T.ptx`` calls and can acquire their own
documented pattern separately.

Implementation and extension
--------------------------------------------------------------------------------

``python/tvm/backend/cuda/tile/patterns/`` has one module per fixed instruction family.
``common.py`` evaluates canonical coordinates, validates group/repeat shapes and
physical storage, and emits repeat loops. ``descriptors.py`` extracts and verifies
shared matrix descriptor fields. There are no imports of retired CUDA dispatch
implementations.

To add a pattern:

1. Define its canonical operand suffixes, ownership and physical address walk,
   all semantic qualifiers, and required caller preparation in the corresponding instruction guide.
2. Resolve the exact instruction through ``ptx.instruction_forms``. Reuse the
   existing PTX table for legality and native operand emission.
3. Use ``Context`` and ``Region`` to check the supplied views at those fixed
   coordinates. Reject mismatches; do not search for another layout or opcode.
4. Emit one selected instruction inside ``Context.finish``. Put algorithms,
   packing, synchronization and thread election at the call site.
5. Add positive/negative contracts, script/JSON roundtrips, exact PTX spelling
   and issue-count checks, and GPU results reconstructed independently from
   the documented hardware mapping.

The common IR is ``Call(opcode, operands, parameters, ty=ty)`` with only execution scope in
``ScopeAttrs``. Runtime and static descriptor parameters both live in ordinary
arguments. ``ewise`` stores its typed scalar body as ``LambdaExpr``. No process-local
Python callable survives serialization.

Region extents are positive static integers. Starts may be symbolic when their bounds
and alignment can be proved. General physical-equivalence discovery is outside the
contract. Gapped allocation covers the layout's physical span, including the address
range introduced by a composed swizzle.
