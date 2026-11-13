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

cp.async
========

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_shared, src_global``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.
   * - ``chunk_bytes``
     - static int; required
     - Bytes per atom: 4, 8 or 16, subject to the chosen PTX form (.cg uses 16).
   * - ``src_size``
     - native byte-count operand; default None
     - Copy this prefix and zero-fill the remainder; mutually exclusive with ignore_src. The selected PTX form determines immediate restrictions.
   * - ``ignore_src``
     - runtime bool; default None
     - True zero-fills the destination; mutually exclusive with src_size.
   * - ``cache_policy``
     - runtime uint64; required with .L2::cache_hint
     - Native cache-policy operand.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
-------------------

Each operand ends in chunk_bytes*8/storage_bits scalar elements. Source and destination
may use different equal-total-size storage representations; this is a byte copy.

Operand specification
---------------------

``cp.async`` copies a short contiguous chunk from global to shared memory without
passing the payload through registers. Each issuing thread owns its async-copy group.
Select the cache/prefetch qualifiers in the opcode and the copy width with
``chunk_bytes``::

    T.cuda.tile.apply(
        T.ptx["cp.async.cg.shared.global"], dst, src,
        chunk_bytes=16, scope="cta",
    )

Operand contract
~~~~~~~~~~~~~~~~

The operand order is ``dst_shared, src_global``. At thread scope each region has shape
``(*repeat,element)``. The final axis occupies exactly ``chunk_bytes`` bytes and is
contiguous and naturally aligned on both sides. For float16 with ``chunk_bytes=16``,
that axis has eight elements. An explicit b32 view of the same bytes has four elements.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``.ca`` / ``.cg``
     - Select in the opcode. ``.ca`` supports 4/8/16 bytes; ``.cg`` requires 16.
   * - ``chunk_bytes``
     - Required constant 4, 8 or 16; no width is inferred from the layout.
   * - ``.L2::64B`` / ``.L2::128B`` / ``.L2::256B``
     - Optional prefetch qualifier in the opcode, not a larger copy size.
   * - ``cache_policy``
     - Runtime operand for an opcode containing ``.L2::cache_hint``.
   * - ``scope``
     - Thread by default. CTA scope adds explicit ``(warp,lane)`` group axes.

The same byte count is copied when source and destination use different storage element
widths, provided each side's atom contains that many whole contiguous bytes. This is a
bit copy, not a conversion.

Legality conditions
-------------------

Source is global; destination is shared/shared.dyn. Each atom is contiguous and aligned
to chunk_bytes. The chunk contains whole elements on both sides. Group/repeat shapes
agree. Source bounds must describe a valid region even when zero-fill operands are
supplied.

Code generation
---------------

At each repeat, emit the selected cp.async with destination/source atom pointers,
chunk_bytes and the explicit optional zero-fill/cache operands. No commit_group,
wait_group, barrier or thread election is inserted. pred suppresses the issue;
ignore_src/src_size instead issue a copy with zero fill.

Examples
--------

Example: four 16-byte copies per thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This complete kernel uses ``(128,32)`` float16 global and shared tensors. It explicitly
assigns four eight-element chunks to each of 128 threads:

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def copy_async(A: T.Tensor((128, 32), "float16"), B: T.Tensor((128, 32), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        shared = T.alloc_shared((128, 32), "float16", layout="default")
        src = A.rearrange("(w t) (r v) -> w t r v", w=4, t=32, v=8)
        dst = shared.rearrange("(w t) (r v) -> w t r v", w=4, t=32, v=8)
        T.cuda.tile.apply(
            T.ptx["cp.async.cg.shared.global"],
            dst,
            src,
            chunk_bytes=16,
            scope="cta",
        )
        T.ptx.cp.async_.commit_group()
        T.ptx.cp.async_.wait_group(0)
        T.cuda.cta_sync()
        for j in range(32):
            B[tid, j] = shared[tid, j]

The views have shape ``(4,32,4,8)``. Coordinates ``w,t`` bind the warp and lane, ``r``
repeats the instruction four times, and ``v`` selects eight float16 elements. For thread
``tid=32*w+t``, issue ``r`` accesses ``A[tid,8*r:8*r+8]`` and the matching shared row.
There is no data rearrangement.

Compile and check it with an Ampere-or-later CUDA device:

.. code-block:: python

    import numpy as np
    import tvm

    compiled = tvm.tirx.compile(copy_async, target="cuda", pipeline="tirx")
    x = (np.arange(4096) % 97).astype("float16").reshape(128, 32)
    a = tvm.runtime.tensor(x, tvm.cuda())
    b = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    compiled(a, b)
    np.testing.assert_array_equal(b.numpy(), x)

The transfer part emits four instances of this instruction per thread, followed by the
caller's commit and wait. Pointer calculations and inline helper names are omitted here:

.. code-block:: text

    cp.async.cg.shared.global [shared_address], [global_address], 16;
    // repeated for r = 0, 1, 2, 3
    cp.async.commit_group;
    cp.async.wait_group 0;

Predication is different from zero fill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pred=False`` suppresses the instruction; it does not write zeros to shared memory. To
copy a partial chunk and zero the remaining bytes, supply ``src_size`` in the range
``0..chunk_bytes``. To zero the entire chunk, use ``src_size=0`` or ``ignore_src=True``.
``src_size`` and ``ignore_src`` are alternative operand forms and cannot both be
supplied.

For example, inside a kernel where ``tid`` selects one 16-byte row::

    T.cuda.tile.apply(
        T.ptx["cp.async.cg.shared.global.L2::128B"],
        shared.sub[tid, :], A.sub[tid, :], chunk_bytes=16,
        src_size=T.if_then_else(tid < valid_rows, T.uint32(16), T.uint32(0)),
    )

Here both tensors have uint8 rows of length 16. The final allocated row is still part of
the declared source view: a zero-fill operand does not relax the tile pattern's
bounds/shape proof.

Completion and errors
~~~~~~~~~~~~~~~~~~~~~~

Each thread commits and waits for its own issues. Use a CTA barrier before other threads
read the resulting shared tile. These calls are not inserted by ``apply``. A
shared-to-global request, a noncontiguous chunk, an unaligned base, or a missing group
axis fails its contract; lowering never substitutes a synchronous copy or a narrower
instruction.

Implementation: ``python/tvm/backend/cuda/tile/patterns/async_copy.py``. See
:doc:`../cuda_tile_patterns` for the shared group/repeat rules.
