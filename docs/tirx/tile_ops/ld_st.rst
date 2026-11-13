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

ld / st
========

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_registers, src_memory for ld; dst_memory, src_registers for st``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.
   * - ``cache_policy``
     - runtime uint64; required with .L2::cache_hint
     - Native cache-policy operand; absent on forms without that operand.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional.
   * - ``preserve_dst``
     - static bool; default False
     - For a predicated register result, retain the initialized destination when pred is false; otherwise the inactive result is undefined.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
-------------------

Both operands use scalar storage elements with the opcode's bit width. Vector length V
is taken from .v2/.v4 or another supported vector qualifier; scalar forms use V=1.

Operand specification
---------------------

Use ``T.cuda.tile.apply`` to repeat a chosen memory instruction. The operand order is
always destination first: a load takes ``registers, memory`` and a store takes ``memory,
registers``. For example::

    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R, A)
    T.cuda.tile.apply(T.ptx["st.shared.v4.b32"], S, R)

Both calls transfer four 32-bit words per repeat. Global-to-shared copying through
registers consists of these two explicit calls and caller-allocated register storage. To
issue a direct asynchronous transfer, use :doc:`cp_async`.

Contract and instruction selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operands have shape ``(*group,*repeat,word)``. The final extent is the vector length in
the opcode, or one for a scalar instruction. Both operands use scalar elements with the
opcode's storage bit width. Each memory atom must be consecutive and aligned to the
whole vector's byte size. Its start can vary across repeats. Local register slots can be
separated or permuted, but each must belong to the issuing thread; loaded destinations
cannot overlap.

The supported address-space spellings are ``global``, ``shared``, ``shared::cta``, and
``local``. The pointer side of ``ld.local``/``st.local`` is addressable local memory,
distinct from the register operands. Legal types, vector lengths, and qualifiers come
from the selected PTX form.

Cache operators such as ``.nc`` and ``.cg`` and prefetch/eviction qualifiers are part of
the opcode. A form with a cache-policy operand takes ``cache_policy=...``. ``pred``
gates execution. A predicated load can use ``preserve_dst=True`` to keep an already
initialized destination when the predicate is false. Lowering does not reduce the width
when alignment fails.

Legality conditions
-------------------

Both regions end in (V,). Memory elements within an atom are consecutive, and the atom
start is aligned to V*element_bytes. The written address space must match storage. Local
register slots belong to the issuing thread and loaded destinations do not overlap. A
region start participates in every address and alignment check.

Code generation
---------------

For each repeat, take the memory pointer at word zero and bind the V register slots in
word order. Emit one selected ld/st, optionally with cache_policy and pred. Repeat axes
unroll; group axes bind hardware IDs. Ordinary ld/st performs no asynchronous completion
protocol.

Examples
--------

Example: a strided tensor and gapped registers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following kernel reads a float32 tensor whose logical axes are not in physical
order. It keeps that layout and prepares a view with four physically adjacent words at
the end. The destination deliberately has twice the storage stride, so each vector
writes separated local slots.

.. code-block:: python

    from tvm.script import tirx as T
    from tvm.tirx.layout import S, TileLayout

    global_layout = TileLayout(S[(15, 2, 7, 2) : (4, 1, 60, 2)])
    register_layout = TileLayout(S[(15, 2, 7, 2) : (8, 2, 120, 4)])


    @T.function
    def strided_load(
        A: T.Tensor((15, 2, 7, 2), "float32", layout=global_layout, align=16),
        Out: T.Tensor((420,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        R = T.alloc_tensor(
            ty_args=[T.Tensor((15, 2, 7, 2), "float32", scope="local", layout=register_layout)]
        )
        av = A.rearrange("i b j d -> j i (d b)")
        rv = R.rearrange("i b j d -> j i (d b)")
        T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], rv, av)
        for j, i, d, b in T.grid(7, 15, 2, 2):
            Out[60 * j + 4 * i + 2 * d + b] = R[i, b, j, d]

Both views have shape ``(7,15,4)``. For one repeat ``(j,i)``, source word ``w`` lives at
``60*j+4*i+w`` and destination word ``w`` lives at local offset ``120*j+8*i+2*w``. One
instruction binds its four outputs directly to these four slots. This emits **105
ld.global.v4.b32 instructions**, with no intermediate tile or register-remapping copy.
The final scalar stores are ordinary caller code used to inspect the result.

Compile and check the complete kernel:

.. code-block:: python

    import numpy as np
    import tvm

    compiled = tvm.tirx.compile(strided_load, target="cuda", pipeline="tirx")
    x = np.arange(420, dtype="float32")
    a = tvm.runtime.tensor(x.reshape(15, 2, 7, 2), tvm.cuda())
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    compiled(a, out)
    np.testing.assert_array_equal(out.numpy(), x)

The raw load in the generated PTX has four output operands, regardless of the local
offsets selected by the view:

.. code-block:: text

    ld.global.v4.b32 {r0, r1, r2, r3}, [address];

Sharing the work across threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At CTA scope, prepend ``(warp,lane)``. With 128 threads, ``(4,32,3,4)`` means three
four-word instructions per thread. A register layout can assign the first two axes
strides ``1 @ warpid`` and ``1 @ laneid``, and the last two ordinary local strides
``4,1``. Memory views must expose the same group/repeat coordinates. These axes
distribute the operation; they are not searched for in an arbitrary layout.

Loads and stores do not insert a barrier. If another thread will consume shared stores,
synchronize at that dependency. To change ownership or reorder shared data in place,
finish the old reads before overwriting bytes.

Implementation: ``python/tvm/backend/cuda/tile/patterns/memory.py``. See
:doc:`../cuda_tile_patterns` for group, repeat, and bounds rules.
