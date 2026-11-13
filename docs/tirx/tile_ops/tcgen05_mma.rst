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

tcgen05.mma
===========

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``D, A, B; plus SFA, SFB for block_scale``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.
   * - ``instruction_shape``
     - static (M,N,K); required
     - Native instruction shape; M/N are totals for the CTA group.
   * - ``a_major, b_major``
     - static "k" or "mn"; default "k"
     - Instruction operand major order; TMEM A requires k.
   * - ``a_format, b_format``
     - static dtype str; default respective tensor dtype
     - Numeric representation, including explicit tf32 over float32.
   * - ``neg_a, neg_b``
     - static bool; default False
     - Instruction descriptor negation bits, subject to the selected kind.
   * - ``accum``
     - runtime bool; default False
     - Use existing D on the first K repeat; subsequent K repeats accumulate.
   * - ``sat_d``
     - static bool; default False; dense forms only
     - Instruction descriptor saturation bit.
   * - ``disable_output_lane``
     - runtime integer tuple; default all zeros; dense non-.ws forms
     - Four mask words per CTA.
   * - ``zero_column_mask``
     - runtime integer; default 0; dense .ws forms
     - Native zero-column mask descriptor operand.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
-------------------

D/A/B retain their documented native TMEM/shared representations. D uses b32 cells even
with 16-bit accumulators; shared values use their storage bit width. Format overrides
affect numeric interpretation, not storage layout. SFA/SFB are 8-bit scale formats in
their native TMEM image.

Operand specification
---------------------

Choose dense, weight-stationary ``.ws``, or ``.block_scale`` in the PTX opcode, together
with CTA group and kind. Supply ``instruction_shape=(M,N,K)`` explicitly. M is the total
for the CTA group, so M=128 with ``cta_group::2`` gives 64 local M rows per CTA. The
instruction is issued by an explicitly elected thread; group scope does not elect one.

Operands and repeat axes
~~~~~~~~~~~~~~~~~~~~~~~~

Dense/.ws calls take ``D,A,B``. Block scale adds ``SFA,SFB``. After any execution-group
and batch prefix, the canonical shapes are:

.. list-table::
   :header-rows: 1

   * - Operand
     - Shape, with g = CTA group
     - Storage
   * - D
     - ``(M_repeat,N_repeat,M/g,N)``
     - TMEM
   * - A
     - ``(M_repeat,K_repeat,M/g,K)``
     - Shared or TMEM
   * - B
     - ``(N_repeat,K_repeat,N/g,K)``
     - Shared
   * - SFA
     - ``(M_repeat,K_repeat,M/g,scale_vec)``
     - TMEM
   * - SFB
     - ``(N_repeat,K_repeat,N,scale_vec)``
     - TMEM, including full N in each CTA of a group-2 operation

M64 .ws with TMEM A adds an atom bank axis: A ends in ``(2,64,K)`` rather than
``(64,K)``. D and A use the selected native TMEM datapath; the pure
``tmem_mma_operand_layout`` helper constructs that layout from explicit shape/type/mode
parameters. It does not choose the MMA opcode.

D stores one accumulator per b32 TMEM cell. Float16 accumulation uses only the low half,
hence a TCol stride of two in 16-bit element units. A in TMEM is packed according to its
input format instead. ``a_format``/``b_format`` default to tensor dtype; supply
``"tf32"`` explicitly for TF32 instructions whose storage dtype is float32.

Shared descriptors are byte-stride contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared layouts determine each operand's LDO/SDO/swizzle. No descriptor tuple is
passed. ``a_major``/``b_major`` are explicitly ``"k"`` or ``"mn"``. For swizzle modes
0..3, a descriptor row is ``16 << swizzle`` bytes and groups contain eight rows. K-major
rows enumerate M/N; MN-major rows enumerate K. The next row group and next contiguous
chunk use:

.. list-table::
   :header-rows: 1

   * - Mode
     - Row-group stride
     - Contiguous-chunk stride
   * - K-major
     - SDO
     - LDO; swizzled instruction atoms fit within one row
   * - MN-major, swizzled
     - SDO
     - LDO
   * - MN-major, unswizzled
     - LDO
     - SDO

The final physical address is the selected swizzle of this byte walk. The origin is
inverse-swizzled before encoding so it is not permuted twice. Descriptor strides are
16-byte multiples below 2^18, origins are 16-byte aligned, and backing allocations align
to the full swizzle period. Fields are extracted from fixed canonical coordinates and
the full address walk is checked.

Descriptor mode 4 is the separate 128B-span/32B-atomicity mode for 32-bit MN-major
operands. It groups **four** K rows of 128 bytes per SDO and uses LDO for the next
128-byte M/N chunk. Its permutation is ``offset ^ ((offset >> 2) & 96)``, with a
512-byte period. It is not an alternative row pitch for ordinary mode 3, and is not this
pattern's K-major mode. TF32 MN-major requires this mode; other MN-major element widths
use modes 0..3.

Legality conditions
-------------------

Instruction kind, shape, formats, major order, CTA group and scale vector must agree.
Shared operands match an extracted descriptor; TMEM operands match the native datapath
and replica map. All group/batch/repeat extents are static. Descriptor strides fit 16B
units below 2^18; origins and backing allocations meet descriptor alignment. M64 .ws
A-in-TMEM has the explicit bank axis described in the operand specification.

Code generation
---------------

Expand batch, M_repeat, N_repeat, K_repeat in that order. Read D at (m,n), A at (m,k), B
at (n,k), and the corresponding scale images. Extract each shared descriptor; encode
instruction_shape, formats, major and flags into the instruction descriptor. Emit one
selected MMA per repeat tuple. Its accumulation predicate is accum OR k_repeat!=0. No
implicit K decomposition, staging, election, commit or wait is inserted.

Examples
--------

Example: four K16 issues for a 128x64x64 product
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input B is stored as ``(N,K)``. Shared initialization uses ordinary scalar stores so
the layout and MMA behavior can be inspected independently of a TMA pipeline. The
operand layouts and explicit K decomposition determine the descriptor fields and issues.

.. code-block:: python

    from tvm.script import tirx as T
    from tvm.backend.cuda.layout import mma_shared_layout, tmem_mma_operand_layout

    al = mma_shared_layout("float16", 3, (128, 64))
    bl = mma_shared_layout("float16", 3, (64, 64))
    dl = tmem_mma_operand_layout("D", (128, 64), "float32", M=128, cta_group=1)


    @T.function
    def tensor_mma(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((64, 64), "float16"),
        Out: T.Tensor((128, 64), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        a = T.alloc_shared((128, 64), "float16", layout=al, align=1024)
        b = T.alloc_shared((64, 64), "float16", layout=bl, align=1024)
        address = T.alloc_shared((1,), "uint32", layout="default")
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(64)
            )
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        for j in range(64):
            a[tid, j] = A[tid, j]
            if tid < 64:
                b[tid, j] = B[tid, j]
        T.cuda.cta_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        d = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((128, 64), "float32", scope="tmem", layout=dl)]
        )
        av = a.rearrange("m (kr k) -> kr m k", k=16).view(1, 4, 128, 16)
        bv = b.rearrange("n (kr k) -> kr n k", k=16).view(1, 4, 64, 16)
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                d.view(1, 1, 128, 64),
                av,
                bv,
                instruction_shape=(128, 64, 16),
                accum=False,
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        frag = T.alloc_tensor(ty_args=[
            T.Tensor((128, 64), "float32", scope="local",
                     layout=T.TileLayout(T.S[(128, 64) : (1 @ T.cuda.Axis.tid_in_wg, 1)]))
        ])
        T.cuda.tile.apply(
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"],
            frag.view(4, 32, 64),
            d.view(4, 32, 64),
            scope="warpgroup",
        )
        T.ptx.tcgen05.wait__ld.sync.aligned()
        local = frag.view(64, layout="default")
        for j in range(64):
            Out[tid, j] = local[j]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(64))

.. code-block:: python

    import numpy as np
    import tvm

    rng = np.random.default_rng(0)
    a = (rng.integers(-4, 5, (128, 64)) / 4).astype("float16")
    b = (rng.integers(-4, 5, (64, 64)) / 4).astype("float16")
    compiled = tvm.tirx.compile(tensor_mma, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros((128, 64), dtype="float32"), tvm.cuda())
    compiled(tvm.runtime.tensor(a, tvm.cuda()), tvm.runtime.tensor(b, tvm.cuda()), out)
    np.testing.assert_allclose(
        out.numpy(), a.astype("float32") @ b.astype("float32").T, rtol=2e-4, atol=2e-4
    )

The issuing thread emits four identical-kind MMA instructions, one for each explicit K
repeat. ``accum=False`` applies to the first K issue only; later issues enable D input.
One commit covers that sequence. The four warps then each execute one x64 load and wait
before consuming their registers.

Weight-stationary and block scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``.ws`` is selected in the opcode, not inferred from packed C. M64 .ws uses Layout E
with an explicit bank dimension for TMEM A; M128 .ws uses Layout D. The current pattern
requires CTA group 1 and N=64/128/256 for .ws. Splitting an M64 N128 operation into two
N64 issues changes each instruction's bank boundary from 64 to 32 columns. Preserve the
intended result with explicit B preparation and a matching C view; cutting two
consecutive logical N slices is insufficient. ``test_tcgen05_mma_ws_split_n.py`` gives
the complete numerical example with distinct data in the two A banks.

Block-scaled opcodes specify ``.kind`` and ``.scale_vec::1X/2X/4X``. Prepare SFA/SFB in
TMEM: scale row r uses lane ``r%32`` and byte-column ``4*(r//32)+sf``, with four
replicas spaced 32 lanes apart. Scale-factor IDs are encoded from each tensor's own
byte-column origin. Supplying scales neither allocates them nor changes the explicit
input kind/scale-vector mode.

``neg_a``, ``neg_b``, ``sat_d``, dense ``disable_output_lane`` masks,
``zero_column_mask`` for .ws, and ``pred`` are explicit parameters where applicable. The
tile does not initialize barriers, elect a thread, perform TMA, prepare scale factors,
commit or wait. Keep the shared/TMEM allocations alive until their consumers complete.

Implementation: ``patterns/tcgen05_mma.py`` and ``patterns/descriptors.py`` under
``python/tvm/backend/cuda/tile``.
