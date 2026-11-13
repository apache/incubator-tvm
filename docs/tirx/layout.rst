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

Tensor Layout
=============

A tensor layout describes how a logical tensor is stored in physical resources.
TIRx generalizes the classical *shape–stride* model: strides are semantically
**named** and bound to **axes** that represent hardware resources — memory,
threads, and devices. A layout maps each logical index to a *set* of coordinates
on these named axes, decomposed into shard (``D``), replica (``R``), and offset
(``O``).

Interactive demo
----------------

Pick a preset, edit the logical shape and the ``S/R/O`` layout, choose a dtype +
swizzle mode, then click an element to see exactly which physical thread(s) own
it.

.. raw:: html

   <p>
     <a class="reference external" href="https://mlc.ai/modern-gpu-programming-for-mlsys/_static/tirx-layout-demo/index.html"
        target="_blank" rel="noopener"
        style="display:inline-block; padding:10px 18px; background:#3b82f6;
        color:#fff !important; font-weight:700; border-radius:8px;
        text-decoration:none;">▶ Open the interactive layout demo ↗</a>
   </p>

TileLayout
----------

An **iter** is a triple ``(extent, stride, axis)`` that defines a linear,
strided access on one axis.

- **D (Shard).** A list of one or more iters, each with an extent and a stride on
  some axis. ``D`` partitions the logical index across these iters and produces a
  base coordinate; this generalizes shape–stride to multiple axes. Written in
  parentheses, e.g. ``S[(8,2,4,2):(4@laneid,1@warpid,1@laneid,1)]``.
- **R (Replica).** A set of replication iters that enumerate offsets in hardware
  space, independent of the logical index. Adding each element of the set to the
  ``D`` result yields replication or broadcasting. Written in square brackets,
  e.g. ``R[2:4@warpid]``.
- **O (Offset).** A fixed coordinate offset (one integer per axis) added to every
  result. This places data at a base position or reserves exclusive resources.

Formally, for a logical index ``x`` the layout produces

.. math::

   L(x) = \{\, D(x) + r + O \mid r \in R \,\},

where ``D(x)`` is the base coordinate from the sharded iters, ``r`` ranges over
all combinations of the replica iters (a single zero offset when ``R`` is empty),
and ``O`` is the constant offset. ``L(x)`` can be a singleton or contain multiple
coordinates. A term is written ``n @ axis``; if a stride is not paired with an
axis, the memory axis ``m`` is used by default.

Forward mapping
~~~~~~~~~~~~~~~

Evaluating ``L(x)`` for a logical coordinate ``x = (x_0, …, x_{r-1})`` in a shape
``(S_0, …, S_{r-1})`` is four mechanical steps.

**1. Flatten** the coordinate row-major to a single index:

.. math::

   \mathrm{flat} = \sum_{d} x_d \prod_{e > d} S_e .

**2. Split** that index across the shard extents ``(e_0, …, e_{n-1})``
(one component ``c_k`` per shard iter, innermost-first):

.. math::

   c_k = \left\lfloor \mathrm{flat} \,\Big/ \textstyle\prod_{l > k} e_l \right\rfloor
         \bmod e_k .

**3. Accumulate** each component onto its axis with its stride to get the base
coordinate, then **add the offset**:

.. math::

   D(x)[a] = \sum_{k\,:\,a_k = a} c_k\, s_k ,
   \qquad
   \bigl(D(x) + O\bigr)[a] = D(x)[a] + O[a] .

**4. Broadcast** the replica iters: ``r`` ranges over
``∏_t [0, e_t)`` and adds, per replica iter ``(e_t, s_t, a_t)``, ``r_t s_t`` to
axis ``a_t`` — yielding the set ``L(x)``:

.. math::

   L(x)[a] = D(x)[a] + O[a] + \sum_{t\,:\,a_t = a} r_t\, s_t .

When a shape's total size equals ``∏_k e_k``, flattening and splitting covers
the complete shard coordinate space exactly once; the same layout can therefore
be reshaped to any dimensions with that total size. Choose the tensor shape
and layout together so logical indices stay within the intended shard
coordinate space.

Case study: NVIDIA tensor-core tile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a logical ``(8, 16)`` tile distributed across 2 warps of 32 lanes each,
with each lane holding part of the tile in its registers (the ``reg`` slot is the
default memory axis ``m``)::

    S[(8,2,4,2):(4@laneid,1@warpid,1@laneid,1)] + R[2:4@warpid] + 5@warpid

The shard factors the logical indices into four iters of extent ``8, 2, 4, 2``
over axes ``laneid, warpid, laneid, m``. Running the four steps on element
``(i, j)``:

#. **flatten:** ``flat = 16 i + j``.
#. **split** by ``(8, 2, 4, 2)``: ``c_0 = i``, ``c_1 = ⌊j/8⌋``,
   ``c_2 = ⌊j/2⌋ mod 4``, ``c_3 = j mod 2``.
#. **accumulate + offset:** ``laneid = 4 c_0 + c_2 = 4 i + ⌊j/2⌋ mod 4`` (two
   iters land on ``laneid``); ``warpid = c_1 + 5 = ⌊j/8⌋ + 5`` (offset
   ``5@warpid``); ``m = c_3 = j mod 2``.
#. **replica** ``R[2:4@warpid]``: ``r ∈ {0, 1}`` adds ``4r`` to ``warpid``, so
   each element lives on **two** warps.

So the full mapping is

.. math::

   \mathrm{laneid} = 4 i + \lfloor j/2 \rfloor \bmod 4, \quad
   \mathrm{warpid} = \lfloor j/8 \rfloor + 5 + 4 r\ (r \in \{0,1\}), \quad
   m = j \bmod 2 .

The shard places the tile on warps ``{5, 6}`` (``⌊j/8⌋ + 5``); the replica copies
it to ``{9, 10}``. A few elements:

.. list-table::
   :header-rows: 1
   :widths: 14 10 26 12 20 10

   * - ``(i, j)``
     - flat
     - ``(c0, c1, c2, c3)``
     - laneid
     - warpid (×2)
     - ``m``
   * - ``(0, 0)``
     - 0
     - ``(0, 0, 0, 0)``
     - 0
     - ``{5, 9}``
     - 0
   * - ``(0, 1)``
     - 1
     - ``(0, 0, 0, 1)``
     - 0
     - ``{5, 9}``
     - 1
   * - ``(0, 2)``
     - 2
     - ``(0, 0, 1, 0)``
     - 1
     - ``{5, 9}``
     - 0
   * - ``(1, 0)``
     - 16
     - ``(1, 0, 0, 0)``
     - 4
     - ``{5, 9}``
     - 0
   * - ``(0, 8)``
     - 8
     - ``(0, 1, 0, 0)``
     - 0
     - ``{6, 10}``
     - 0
   * - ``(7, 15)``
     - 127
     - ``(7, 1, 3, 1)``
     - 31
     - ``{6, 10}``
     - 1

Case study: Blackwell tensor memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same machinery places a tensor into Blackwell **tensor memory**, a 2D address
space addressed by ``TLane`` × ``TCol`` (both *memory* axes). Where the register
tile bound strides to thread axes, here every stride binds to a memory axis, so
the layout is a pure placement — no threads, no replica, no offset, and ``L(x)``
is a singleton::

    S[(2,128,112):(112@TCol,1@TLane,1@TCol)]

Take the logical tile shape equal to the shard extents, ``(2, 128, 112)`` — then
the split step is the identity (``c_k = x_k``), and element ``(a, l, c)`` maps to:

.. math::

   \mathrm{TLane} = l, \qquad \mathrm{TCol} = 112\,a + c .

The extent-128 iter (``1@TLane``) lays the tile across **128 lanes**; the
extent-2 iter (``112@TCol``) and the extent-112 iter (``1@TCol``) together cover
**224 columns** (``TCol = 112 a + c ∈ [0, 224)``). A few elements:

.. list-table::
   :header-rows: 1
   :widths: 28 14 14

   * - ``(a, l, c)``
     - TLane
     - TCol
   * - ``(0, 0, 0)``
     - 0
     - 0
   * - ``(0, 5, 3)``
     - 5
     - 3
   * - ``(1, 0, 0)``
     - 0
     - 112
   * - ``(1, 127, 111)``
     - 127
     - 223

The 224-wide span is intentionally **not a power of two**: a block-scaled FP8
GEMM may use a 224-column tile because tensor memory cannot hold two accumulator
stages plus the scale factors at 256. General-shape support is what lets the
layout express this directly.

**Scale factors (SFA / SFB).** A block-scaled MMA also keeps its per-block scale
factors in tensor memory, and their layout is the first one here to use a
**replica**. The atom is::

    S[(32, sf_per_mma):(1@TLane, 1@TCol)] + R[4:32@TLane]

— 32 rows on ``TLane`` and ``sf_per_mma`` scale factors on ``TCol``, with the
replica ``R[4:32@TLane]`` routing that 32-row group across the **four warps** of a
warpgroup (stride 32 covers lanes 0–127) — the "warpx4" router, so one physical
scale-factor group feeds all four warps. The atom is then direct-summed with an
outer over ``(M rows, K scale-factor groups)``, packing ``epc = 32 / SF_bits``
scale factors into each 32-bit ``TCol`` cell (e.g. four fp8 ``e8m0`` SFs per cell);
optional stride-0 ``reuse`` and outer ``pipe_depth`` iters express SF reuse across
MMAs and double-tensoring. So the one ``TileLayout`` model expresses both the
accumulator (a pure placement, no replica) and its scale factors (a replicated,
routed placement) in the same tensor-memory address space.

TMEM datapath layouts
~~~~~~~~~~~~~~~~~~~~~

``tmem_datapath_layout`` provides the canonical per-CTA placement for all seven
``tcgen05`` MMA datapaths:

.. list-table::
   :header-rows: 1
   :widths: 10 22 23 45

   * - Layout
     - Instruction case
     - Per-CTA logical rows
     - Placement
   * - A
     - M=256, CTA group 2
     - 128
     - identity row-to-lane mapping
   * - B
     - M=128, CTA group 2, dense A
     - 64
     - fold two column halves across two 64-lane halves; even columns required
   * - C
     - M=128, CTA group 2, sparse A
     - 64
     - scatter 16 rows into each warp's 32-lane slab
   * - D
     - M=128, CTA group 1
     - 128
     - identity row-to-lane mapping
   * - E
     - M=64, ``.ws``
     - 64
     - the same column-half folding as B; even columns required
   * - F
     - M=64, non-``.ws``
     - 64
     - the same 16-row-per-warp scattering as C
   * - G
     - M=32, ``.ws``
     - 32
     - fold four column quarters across four 32-lane slabs; columns must be
       divisible by four

The examples below show D, F, and B, the layouts used by the current copy
walkthroughs:

.. code-block:: python

    from tvm.tirx.layout import tmem_datapath_layout

    accum = tmem_datapath_layout("D", 128, cols)
    lower = tmem_datapath_layout("F", 64, cols, sub_slab=0)
    upper = tmem_datapath_layout("F", 64, cols, sub_slab=1)
    paired = tmem_datapath_layout("B", 64, cols)

Layout D maps logical row ``r`` directly to ``TLane = r`` and spans both
16-lane halves of every warp's 32-lane TMEM partition. Layouts C and F map
their 64 logical rows according to

.. math::

   \mathrm{TLane}
   = 32\left\lfloor\frac{r}{16}\right\rfloor
     + 16\,\mathrm{sub\_slab}
     + (r \bmod 16).

Thus C/F with ``sub_slab=0`` and ``sub_slab=1`` can describe lower- and
upper-half aliases of the same 128-row Layout D allocation. The
explicit TMEM pattern validates the selected row slab and emits its
``row=0`` or ``row=16`` address. Layout D already occupies both halves,
so a nonzero ``sub_slab`` is rejected.

Layout B is the per-CTA accumulator placement for an M=64
``tcgen05.mma.cta_group::2`` operation. PTX describes the two-CTA operation
as M=128, while each CTA owns a logical ``(64, N)`` tile. Its columns split
across the two 64-lane halves:

.. math::

   \mathrm{TLane}
   = r + 64\left\lfloor\frac{c}{N/2}\right\rfloor,
   \qquad
   \mathrm{TCol} = c \bmod (N/2).

Thus the tile occupies all 128 lanes and ``N/2`` tensor-memory columns, the
same physical footprint as Layout D ``(128, N/2)``. ``N`` must be even and
Layout B does not accept ``sub_slab``. Its logical
``(64, N)`` register fragment has ``N/2`` values per thread. To load it,
expose the physical slab as ``(4, 32, N/2)``: the leading axis selects a
warp, and the remaining axes describe rows and words. Choose an explicit
``tcgen05.ld.32x32b.xN`` width and split the word axis into repetitions of
that width. The matching register and TMEM patterns are documented in
:doc:`cuda_tile_patterns`; the caller issues ``tcgen05.wait::ld`` before
using the loaded registers.

Beyond GPU registers
~~~~~~~~~~~~~~~~~~~~~~

The same layout describes more than register tiles. Binding strides to a device
axis (``pid``) expresses **distributed sharding** across a GPU mesh; binding them
to on-chip memory axes expresses native accelerator memories — a 2D-partitioned
scratchpad (partition ``P`` and free ``F`` axes), or NVIDIA Blackwell tensor
memory with native 2D addressing (``TLane`` × ``TCol``). The demo includes
presets for each.

ComposeLayout (swizzled tile)
-----------------------------

Some layouts also need a *swizzle*: a non-linear, XOR-based permutation of the
linear memory address. It is not expressible as a strided ``TileLayout`` (which
is affine), so TIRx folds it into a ``ComposeLayout``: alongside a
``tile_layout``, a ``ComposeLayout`` carries four swizzle parameters
(``per_element``, ``swizzle_len``, ``atom_len``, ``swizzle_inner``). The tile
layout produces a linear memory address; the swizzle then permutes that address.
A *bare* swizzle (no meaningful tile) is a ``ComposeLayout`` over a trivial
identity ``TileLayout`` covering one swizzle period.

Why swizzle
~~~~~~~~~~~

Shared memory is organized into **32 banks of 4 bytes**. Consecutive 4-byte
words land in consecutive banks and wrap every ``32 × 4 = 128`` bytes (one
*bank line*). A bank conflict occurs when the threads of an access touch
different addresses in the **same** bank.

Store a tile row-major and the conflict is structural. Take an ``(8, 64)``
``float16`` tile (``S[(8,64):(64@m,1@m)]`` — element ``(i, j)`` at address
``m = 64i + j``). One row is ``64 × 2 = 128`` bytes = exactly one bank line, so
walking *down a column* (fixed ``j``, increasing ``i``) jumps the address by a
whole bank line and lands on the **same bank** every time — an 8-way conflict
for this eight-row example. Swizzle scatters those accesses across banks.

The transform
~~~~~~~~~~~~~

The swizzle has three integer parameters — ``per_element`` (M),
``swizzle_len`` (B), and ``atom_len`` (S) — and maps a linear element address
``m`` as follows (keeping the low ``M`` bits untouched and XOR-ing a higher bit
group down into a lower one; this is the ``swizzle_inner=True`` direction, and
``swizzle_inner=False`` mirrors the XOR):

.. math::

   \text{addr}(m) = \bigl(f(m \gg M)\bigr)\!\cdot\! 2^{M} + (m \bmod 2^{M}),
   \qquad
   f(x) = x \oplus \bigl((x \mathbin{\&} (\,(2^{B}-1)\ll S\,)) \gg S\bigr).

So the bits at positions ``[S, S+B)`` of ``x = m >> M`` are XOR-ed into bits
``[0, B)``. The well-formedness requirement is ``S ≥ B``.

Choosing the parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

In practice the parameters come from the **element dtype** and the **swizzle
mode** (the 32B / 64B / 128B shared-memory swizzle widths):

.. math::

   M = \operatorname{bitlen}\!\left(\frac{128}{\text{dtype bits}}\right) - 1,
   \qquad
   B = \begin{cases} 1 & 32\text{B} \\ 2 & 64\text{B} \\ 3 & 128\text{B} \end{cases},
   \qquad
   S = 3 .

For example ``float16`` (16-bit) gives ``M = bitlen(8) - 1 = 3``; with 128B
swizzle that is swizzle ``(M=3, B=3, S=3)``. ``M`` keeps a 16-byte (128-bit)
contiguous run unswizzled, matching the minimum vector access.

Bank and line of an element
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because a bank word is 4 bytes and an element is ``b = dtype_bytes`` bytes, the
swizzled element address ``a = addr(m)`` lands in

.. math::

   \text{bankword} = \left\lfloor \frac{a \cdot b}{4} \right\rfloor,
   \qquad
   \text{bank} = \text{bankword} \bmod 32,
   \qquad
   \text{line} = \left\lfloor \frac{\text{bankword}}{32} \right\rfloor .

Worked example: 128B swizzle, ``float16``, ``(8, 64)`` tile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the swizzle ``(3,3,3)`` over ``m = 64i + j`` the address simplifies to

.. math::

   \text{addr}(i, j) = 64\,i + 8\,(\lfloor j/8 \rfloor \oplus i) + (j \bmod 8),

and (``b = 2``) ``bank = ⌊addr/2⌋ mod 32``. Reading column ``j = 0`` down the
8 rows gives ``addr = 72 i`` and banks ``0, 4, 8, 12, 16, 20, 24, 28`` — **eight
distinct banks, no conflict**. Without the swizzle the same column is
``bank = ⌊j/2⌋`` for every row — a single bank, fully serialized.

The swizzle width controls how many of these rows are separated. For the same
``float16`` column, 32B mode (``B=1``) reaches two distinct banks and leaves a
4-way conflict; 64B mode (``B=2``) reaches four banks and leaves a 2-way
conflict; 128B mode (``B=3``) reaches all eight banks and is conflict-free.
Conflict freedom therefore depends on the access shape, dtype, layout, and
swizzle width; choosing any nonzero swizzle mode is not by itself sufficient.

In the interactive demo, pick a dtype and a swizzle mode (``none`` / ``32B`` /
``64B`` / ``128B``) in the *Swizzle (SMEM)* control. The physical panel switches
to a *line × bank* view (each cell is one 4-byte bank word, holding
``4 / dtype_bytes`` elements side by side): with ``none`` a column maps to one
bank (the conflict); with a swizzle the same column is scattered across banks.

Design rationale
----------------

- **General shape support.** Non-power-of-two shapes are common — in global
  tensors, multi-stage shared-memory tensors, and capacity-limited on-chip
  scratchpads — so the layout supports general shapes directly rather than as a
  special case.
- **Logical-to-physical mapping.** The map goes from logical coordinates to a set
  of physical coordinates. This lets replication (one logical element in multiple
  physical locations) be expressed cleanly, which a physical-to-logical
  formulation cannot always represent for strided patterns.
- **Explicit hardware axes.** Axes carry their hardware meaning in the layout
  itself, so an expression is unambiguous without external context. For instance
  ``1@tid`` (block-wide thread id) and ``1@tid_in_wg`` (thread id within a
  warpgroup) are distinct rather than a generic ``t`` whose meaning depends on the
  definition site. Legality and feasibility checks are left to tile op
  dispatch.
