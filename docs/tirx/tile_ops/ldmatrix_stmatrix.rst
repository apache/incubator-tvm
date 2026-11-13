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

ldmatrix / stmatrix
===================

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_registers, src_shared for ldmatrix; dst_shared, src_registers for
stmatrix``.

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

Both operands use scalar b32 containers (uint32/int32/float32 as bits). Each packs two
b16 matrix elements. x1/x2/x4 selects N=1/2/4 b32 outputs per lane. Two halves of one
output are fixed packing; a free register stride separates distinct b32 outputs.

Operand specification
---------------------

The tile patterns support ``m8n8.b16`` with ``.x1``, ``.x2``, or ``.x4`` and optional
``.trans``. Select the exact form in the opcode::

    T.cuda.tile.apply(
        T.ptx["ldmatrix.sync.aligned.m8n8.x2.shared.b16"],
        registers, shared_words, scope="warp",
    )

``stmatrix`` reverses the operand order: ``shared_words, registers``. The implementation
repeats the specified warp instruction, without picking a matrix count or transpose mode
from the layout.

Address providers and result owners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both regions end in ``(matrix,row,word)=(count,8,4)`` and use scalar 32-bit containers.
A container holds two adjacent 16-bit values. Use a dtype view to expose packed
half/bfloat16 storage as b32 words; this is not a conversion.

* Each shared row has four consecutive words and a 16-byte-aligned start.
  Different rows may have gaps and may be interleaved with other matrices.
* Lane ``8*matrix+row`` provides that row's shared address. For x1/x2,
  the remaining address providers repeat valid row addresses.
* Register word ``(matrix,row,word)`` belongs to lane ``4*row+word``.
  Each lane receives ``count`` words. Their register offsets need not be
  adjacent, and need not have the same order as another tensor declaration.

Thus, the lane supplying a shared address generally differs from the lanes receiving the
values from that row. The contract checks both maps. All 32 lanes must participate,
including those whose addresses are duplicates.

Parameterized layouts
~~~~~~~~~~~~~~~~~~~~~

For one warp and one repeat axis, all storage strides below are in b32 words::

    coordinates: (repeat, matrix, row, word)
    smem: (K,N,8,4):(B,   M,   S,        1)
    reg:  (K,N,8,4):(P@m, R@m, 4@laneid, 1@laneid)

B/M/S separate repeats, matrices and shared rows; P/R separate repeats and outputs in
each thread's register storage. They are read from the layout, not supplied as keywords.
S=16 means 64-byte row pitch; R=2 means every second b32 output slot. These slots do not
specify machine register numbers. The more general row-address contract is::

    byte_address(t,i,j,w) = row_address(t,i,j) + 4*w

Each row_address must be 16-byte aligned; it need not have constant stride across j. A
nonzero region start is included before applying the layout. For .trans the b16 image
satisfies::

    register_image[i,j,2*w+h] = shared_matrix[i,2*w+h,j]

Without .trans, the right-hand side is shared_matrix[i,j,2*w+h].

Legality conditions
-------------------

Both regions end in (N,8,4), after matching group/repeat axes. Each shared row has four
consecutive words and a 16-byte-aligned start. Different rows have independent starts.
Register word (i,j,w) belongs to lane 4*j+w; loaded slots are distinct. All 32 lanes
participate; scope="warp" has no group prefix, and wider scopes add warp groups. No
opcode-specific keyword, including pred, is accepted.

Code generation
---------------

For each repeat t, every lane l executes the following expansion::

    address = &smem[t, (l // 8) % N, l % 8, 0]
    registers = [reg[t, i, l // 4, l % 4] for i in range(N)]
    emit_selected_instruction(registers, address)  # ld; reversed for st

The first 8*N lanes supply the N matrices' row addresses; unused address providers
repeat valid addresses. Each repeat emits one warp-wide .xN instruction. Initialization
and synchronization are caller operations. .trans changes the b16 semantic image, as
specified in the operand specification.

Examples
--------

Example: interleaved shared rows and register slots 0, 2, 8, 10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here the original shared tensor is ``(16,4,2):(8,1,128)``. The first matrix uses even
original rows, the second uses odd rows, and the final original axis selects the second
instruction. The factory below includes shared initialization, synchronization, and
readback for one warp.

.. code-block:: python

    from tvm.script import tirx as T
    from tvm.backend.cuda.axis import laneid
    from tvm.tirx.layout import S, TileLayout

    LDMATRIX = "ldmatrix.sync.aligned.m8n8.x2.shared.b16"

    def matrix_kernel():
        shared = TileLayout(S[(16, 4, 2) : (8, 1, 128)])
        registers = TileLayout(S[(2, 2, 8, 4) : (8, 2, 4 @ laneid, 1 @ laneid)])

        @T.function
        def kernel(A: T.Tensor((16, 4, 2), "uint32"), Out: T.Tensor((32, 12), "uint32")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
            lane = T.cuda.lane_id()
            SM = T.alloc_tensor(
                ty_args=[T.Tensor((16, 4, 2), "uint32", scope="shared", layout=shared, align=16)]
            )
            for q in range(4):
                n = lane + 32 * q
                SM[n // 8, (n // 2) % 4, n % 2] = A[n // 8, (n // 2) % 4, n % 2]
            T.cuda.cta_sync()
            raw = T.alloc_tensor(ty_args=[T.Tensor((12,), "uint32", scope="local", layout="default")])
            for i in T.unroll(12):
                raw[i] = T.uint32(4294966519)  # bit pattern of int32(-777)
            R = T.decl_tensor(
                raw.ptr_to([0]),
                ty_args=[T.Tensor((2, 2, 8, 4), "uint32", scope="local", layout=registers)],
            )
            S_std = SM.rearrange("(r m) w k -> k m r w", r=8, m=2)
            T.cuda.tile.apply(T.ptx[LDMATRIX], R, S_std, scope="warp")
            for i in T.unroll(12):
                Out[lane, i] = raw[i]

        return kernel

``S_std[k,m,r,w]`` is ``SM[2*r+m,w,k]`` and has strides ``(128,8,16,1)``. Both operands
have shape ``(2,2,8,4)``: one repeat axis of length two, then the x2 matrix atom. For
lane ``l`` the loaded values are::

    raw[0]  = SM[2*(l//4),   l%4, 0]
    raw[2]  = SM[2*(l//4)+1, l%4, 0]
    raw[8]  = SM[2*(l//4),   l%4, 1]
    raw[10] = SM[2*(l//4)+1, l%4, 1]

All other local slots retain their initialized sentinel. The transfer emits
**two x2 instructions per warp**. It does not emit x4 and does not copy from
a contiguous temporary fragment into the separated output slots.

Run the complete factory example on an Ampere-or-later device:

.. code-block:: python

    import numpy as np
    import tvm

    compiled = tvm.tirx.compile(matrix_kernel(), target="cuda", pipeline="tirx")
    x = np.arange(128, dtype="uint32").reshape(16, 4, 2)
    out = tvm.runtime.tensor(np.zeros((32, 12), dtype="uint32"), tvm.cuda())
    compiled(tvm.runtime.tensor(x, tvm.cuda()), out)
    expected = np.full((32, 12), 4294966519, dtype="uint32")
    for lane in range(32):
        for k in range(2):
            for matrix in range(2):
                expected[lane, 8 * k + 2 * matrix] = x[2 * (lane // 4) + matrix, lane % 4, k]
    np.testing.assert_array_equal(out.numpy(), expected)

Transpose and stores
~~~~~~~~~~~~~~~~~~~~~

``.trans`` transposes each 8x8 matrix of **16-bit elements**, not a matrix of b32 words.
Register image element ``(row,2*word+half)`` comes from shared element
``(2*word+half,row)``. The register word's lane owner remains ``4*row+word``. Reorganize
operands with this image in mind; toggling ``.trans`` is a semantic change, not a layout
optimization hint.

``stmatrix`` uses the corresponding register image as input and each provider's shared
row address as output. Its supported architecture is checked by the PTX compiler.
Synchronize before a load consumes another thread's shared stores, and after a store
before other threads consume or reuse that shared storage. No fence or barrier is
inserted by the pattern.

An insufficiently aligned row, wrong register owner, overlapping loaded registers, or a
partially active warp is rejected. Gaps **between** rows or register slots are legal;
gaps **within** a shared 16-byte row are not.

Implementation: ``python/tvm/backend/cuda/tile/patterns/ldmatrix.py``.
