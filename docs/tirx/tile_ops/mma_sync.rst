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

mma.sync
========

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``D, A, B, C``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
-------------------

D/C use f32 accumulator elements. A/B use b32 containers packing two f16/bf16 values.
The supported tile forms are m16n8k8 and m16n8k16 with f32 accumulation; opcode fixes
the formats and row/column interpretation.

Operand specification
---------------------

The warp tile pattern supports the explicit
``mma.sync.aligned.m16n8k{8,16}.row.col.f32.{f16,bf16}.{f16,bf16}.f32`` forms registered
by the PTX dialect. Each issue multiplies fixed register fragments and accumulates into
four float32 registers per lane. The tile layer does not select K=8 versus K=16, change
input type, load shared memory, or transpose an operand implicitly.

Operand coordinates and ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The call takes ``D,A,B,C``. After optional execution-group and batch axes, the canonical
shapes are:

.. list-table::
   :header-rows: 1

   * - Operand
     - Shape
     - Storage
   * - D / tensor C
     - ``(M_repeat,N_repeat,16,8)``
     - float32
   * - A
     - ``(M_repeat,K_repeat,16,k/2)``
     - b32 words containing two adjacent f16/bf16 K values
   * - B
     - ``(N_repeat,K_repeat,8,k/2)``
     - The same packed representation

B's logical atom axes are ``(N,K/2)``, even though the mathematical product uses B
transposed. For lane ``l``, native register ``r`` corresponds to:

.. list-table::
   :header-rows: 1

   * - Operand
     - Atom coordinate
     - Registers per lane
   * - A
     - ``(l//4+8*(r%2), l%4+4*(r//2))``
     - k/4
   * - B
     - ``(l//4, l%4+4*r)``
     - k/8
   * - D / C
     - ``(l//4+8*(r//2), 2*(l%4)+r%2)``
     - 4

Local offsets can be strided; lane ownership must match these coordinates. All 32 lanes
execute the warp instruction. Warpgroup/CTA scope prepends a warp group axis; that axis
is not an additional matrix dimension.

Legality conditions
-------------------

All tensor operands are local, with the fixed fragment ownership below. D/A/B and tensor
C have matching group/batch and M/N/K repeat extents as applicable. C may be an explicit
scalar broadcast. Writable D slots are nonoverlapping. All 32 warp lanes participate. No
instruction-specific keyword, including pred, is accepted.

Code generation
---------------

Expand batch, M_repeat, N_repeat, K_repeat in that order. Bind each native fragment
using the selected shape's lane/register coordinates. Feed C on K_repeat=0 and the
previous D thereafter, and emit one selected mma.sync. No alpha/beta arithmetic,
implicit fragment conversion or memory staging is inserted.

Examples
--------

Example: a 16x8 result with two K16 issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete kernel below loads the exact packed A/B words each lane needs. The host
prepares these words from ordinary row-major half matrices.

.. code-block:: python

    from tvm.script import tirx as T
    from tvm.backend.cuda.axis import laneid
    from tvm.tirx.layout import S, TileLayout

    la = TileLayout(S[(1, 2, 2, 8, 2, 4) : (8, 4, 1, 4 @ laneid, 2, 1 @ laneid)])
    lb = TileLayout(S[(1, 2, 8, 2, 4) : (4, 2, 4 @ laneid, 1, 1 @ laneid)])
    ld = TileLayout(S[(1, 1, 2, 8, 4, 2) : (4, 4, 2, 4 @ laneid, 1 @ laneid, 1)])


    @T.function
    def warp_mma(
        A: T.Tensor((1, 2, 16, 8), "uint32"),
        B: T.Tensor((1, 2, 8, 8), "uint32"),
        Out: T.Tensor((16, 8), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        ar = T.alloc_local((8,), "uint32", layout="default")
        br = T.alloc_local((4,), "uint32", layout="default")
        dr = T.alloc_local((4,), "float32", layout="default")
        av = ar.view(1, 2, 16, 8, layout=la)
        bv = br.view(1, 2, 8, 8, layout=lb)
        dv = dr.view(1, 1, 16, 8, layout=ld)
        for kk, r in T.grid(2, 4):
            ar[kk * 4 + r] = A[0, kk, lane // 4 + 8 * (r % 2), lane % 4 + 4 * (r // 2)]
        for kk, r in T.grid(2, 2):
            br[kk * 2 + r] = B[0, kk, lane // 4, lane % 4 + 4 * r]
        T.cuda.tile.apply(
            T.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"],
            dv,
            av,
            bv,
            T.float32(0),
            scope="warp",
        )
        for r in range(4):
            Out[lane // 4 + 8 * (r // 2), 2 * (lane % 4) + r % 2] = dr[r]

.. code-block:: python

    import numpy as np
    import tvm

    rng = np.random.default_rng(0)
    a = (rng.integers(-4, 5, (16, 32)) / 4).astype("float16")
    b = (rng.integers(-4, 5, (8, 32)) / 4).astype("float16")
    ap = np.ascontiguousarray(a.reshape(1, 16, 2, 16).transpose(0, 2, 1, 3)).view("uint32")
    bp = np.ascontiguousarray(b.reshape(1, 8, 2, 16).transpose(0, 2, 1, 3)).view("uint32")
    compiled = tvm.tirx.compile(warp_mma, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros((16, 8), dtype="float32"), tvm.cuda())
    compiled(tvm.runtime.tensor(ap, tvm.cuda()), tvm.runtime.tensor(bp, tvm.cuda()), out)
    np.testing.assert_allclose(
        out.numpy(), a.astype("float32") @ b.astype("float32").T, rtol=2e-4, atol=2e-4
    )

``M_repeat=N_repeat=1`` and ``K_repeat=2``, so each warp issues exactly two K16
instructions. The first uses scalar C=0; the second uses the preceding D registers. No
separate zeroing of dr or add instruction is required.

Accumulation, transpose and alpha/beta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repeat order is batch, M, N, then K. C can be a matching register tensor or an explicit
scalar broadcast. Only the first K issue reads the original C; later issues read D. This
is the documented recurrence, rather than an arbitrary traversal inferred from layouts.

To transpose a logical input, prepare a view whose canonical A/B coordinates refer to
the intended values while preserving the required lane map. A view alone cannot exchange
values between lanes; an incompatible physical owner requires explicit preparation.
``alpha``/``beta`` scaling likewise belongs before or after MMA at the call site, with
the intended rounding order.

The tile call inserts no shared loads, ``ldmatrix``, communication, or barrier. Use
:doc:`ldmatrix_stmatrix` to prepare fragments when that native shared/register image
matches your operands. Sparse and other MMA shapes require their own explicit pattern;
this page does not claim those forms.

Implementation: ``python/tvm/backend/cuda/tile/patterns/mma.py``.
