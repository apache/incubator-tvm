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

cp.async.bulk: shared to cluster shared
=======================================

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_shared, src_shared``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.
   * - ``chunk_bytes``
     - static positive int; required
     - Bytes per atom; divisible by 16.
   * - ``mbar``
     - runtime pointer; required
     - Local shared address of the completion barrier in the destination CTA image.
   * - ``remote_cta_id``
     - runtime integer; required
     - Destination CTA rank within the cluster.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
-------------------

Each operand ends in chunk_bytes*8/storage_bits scalar elements. Source, destination and
mbar denote local shared addresses before remote address mapping.

Operand specification
---------------------

This pattern issues
``cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes``. It copies
one contiguous shared-memory chunk to another CTA in the launch cluster and completes
transaction bytes on a barrier in the destination CTA. For global tensor maps, use
:doc:`cp_async_bulk_tensor` instead.

The pointer contract
~~~~~~~~~~~~~~~~~~~~~

``apply`` takes ``dst_shared, src_shared`` followed by explicit ``chunk_bytes``,
``mbar`` and ``remote_cta_id``. Source and destination regions use coordinates
``(*group,*repeat,element)``. The final axis covers exactly ``chunk_bytes``, a positive
multiple of 16; its bytes are consecutive and its origin is 16-byte aligned. Both
shared/shared.dyn storage are accepted.

All three addresses supplied by the caller are **local** shared addresses:

* ``src`` names storage in the issuing CTA.
* ``dst`` names the local counterpart of the destination allocation. Lowering
  maps its address to ``remote_cta_id`` with ``mapa`` for each repeat.
* ``mbar`` names the local counterpart of the completion barrier. Lowering
  maps it to the same remote CTA once before the repeat loop.

Do not pass an already mapped destination or barrier. Both CTAs must have corresponding
live shared allocations at those offsets. ``remote_cta_id`` is the rank within the
cluster, not an absolute block index. ``pred`` gates the copy instruction and must agree
with the caller's transaction accounting.

Legality conditions
-------------------

Both regions use shared/shared.dyn storage, contain whole elements and contiguous
chunks, and have 16-byte-aligned starts. Group/repeat shapes agree. The caller supplies
a valid cluster rank and matching launch/completion protocol.

Code generation
---------------

Map mbar to remote_cta_id once per tile call. In each repeat, map the destination atom
pointer with mapa and emit one cp.async.bulk using the local source, mapped
destination/barrier and chunk_bytes. The core instruction count equals the repeat
product; address preparation includes mapa. Allocation, barrier
initialization/transaction accounting and cluster synchronization belong to the caller.

Examples
--------

Example: CTA 0 sends 256 bytes to CTA 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The one-thread blocks here keep election and initialization visible. The same operation
can be issued by an elected thread in a larger CTA.

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def cluster_copy(A: T.Tensor((64,), "uint32"), Out: T.Tensor((64,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=1, cluster=2))
        rank = T.cuda.cluster_cta_id("x")
        src = T.alloc_shared((64,), "uint32", layout="default", align=16)
        dst = T.alloc_shared((64,), "uint32", layout="default", align=16)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cluster_sync()
        if rank == 0:
            for i in range(64):
                src[i] = A[i]
            T.ptx.fence.proxy.async_.shared__cta()
            T.cuda.tile.apply(
                T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"],
                dst,
                src,
                chunk_bytes=256,
                mbar=mbar.ptr_to([0]),
                remote_cta_id=T.int32(1),
            )
        else:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(256))
            T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
            for i in range(64):
                Out[i] = dst[i]
        T.cuda.cluster_sync()

The sender emits one barrier-address mapa, one destination-address mapa, and one
256-byte bulk copy. The source address stays local. CTA 1 expects 256 transaction bytes
and waits for phase zero before consuming the data. The last cluster barrier keeps both
CTAs' shared allocations alive until the communication is finished.

Compile and verify on a device supporting thread-block clusters:

.. code-block:: python

    import numpy as np
    import tvm

    x = np.arange(64, dtype="uint32")
    compiled = tvm.tirx.compile(cluster_copy, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    compiled(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x)

Repeated chunks and synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A float16 view ``(8,128):(256,1)`` with ``chunk_bytes=256`` represents eight separate
contiguous chunks with gaps between them. It emits eight bulk copies. Set expected
transaction bytes to ``8*256``. The source and destination may have different outer
strides, provided each matching 128-element chunk is contiguous and aligned on its own
side.

Do not flatten across a physical gap and request one larger chunk: it would describe
different bytes and is rejected. There is no automatic split or fallback to a scalar
loop. Cluster launch configuration, issuer election, barrier initialization,
expected-byte accounting, proxy fences, phase progression and storage lifetime are
caller responsibilities. Ordinary ``cp.async.commit_group``/``wait_group`` is not this
completion mechanism.

Implementation: ``python/tvm/backend/cuda/tile/patterns/async_copy.py``.
