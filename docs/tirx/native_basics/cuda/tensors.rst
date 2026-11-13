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

Tensors and memory
==================

``Tx.Tensor`` constructs a ``tirx.TensorType`` for a tensor parameter. Its
shape, dtype, strides, offsets, layout and storage scope describe the same
low-level storage contract used by the allocation and declaration helpers.
Scratch tensors are created in the body with the APIs below. Index a tensor with
``A[i, j]``, slice it with ``A[m0:m0+BM, 0:BK]`` (a ``TensorRegion``), and take a
pointer with ``A.ptr_to([i, j])`` or the raw data pointer ``A.data``.

Declaring tensors
-----------------

The following calls bind tensor variables when assigned in a script function:

- ``Tx.alloc_tensor(ty_args=[Tx.Tensor(shape, dtype, scope=scope)])``
  **allocates new storage**. ``Tx.alloc_shared`` and ``Tx.alloc_local`` remain
  conveniences for constructing a tensor type with shared or local storage.
- ``Tx.decl_tensor(data, ty_args=[Tx.Tensor(shape, dtype, scope=scope)])``
  **declares a view** over an existing pointer ``data`` (no allocation); use it
  to alias or reinterpret storage, for example a sub-region of a pool.
- ``Tx.cuda.decl_tmem(addr, ty_args=[Tx.Tensor(...)])`` — declares a tensor over
  externally allocated tensor memory. The address is a value operand; the tensor
  type supplies shape, dtype, scope and layout.

``A.data`` projects the physical pointer from the tensor variable.
``alloc_tensor`` supplies new storage; ``decl_tensor`` binds an
existing pointer expression. See :doc:`data_types`.

The tensor descriptors share these parameters:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Parameter
     - Meaning
   * - ``dtype``
     - element type — ``"float32"``, ``"float16"``, ``"float4_e2m1fn"``, …
   * - ``shape``
     - logical shape (a tuple of extents)
   * - ``layout``
     - physical mapping, including strides and element origin
       (:doc:`TileLayout <../../layout>`); ``"default"`` = dense row-major
   * - ``align``
     - alignment of the data pointer, in bytes

The ``scope`` argument selects the memory space:

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Scope
     - Shorthand
     - Memory
   * - ``"global"``
     - (default)
     - device global memory
   * - ``"shared"``
     - ``Tx.alloc_shared``
     - static shared memory (``__shared__``)
   * - ``"shared.dyn"``
     - (pool)
     - dynamic shared memory (pooled — see below)
   * - ``"local"``
     - ``Tx.alloc_local``
     - per-thread local storage (usually promoted to registers)
   * - ``"tmem"``
     - (TMEM pool)
     - Blackwell tensor memory (see below)

.. code-block:: python

    @Tx.function
    def kernel(A: Tx.Tensor((M, K), "float16", align=16)):
        As = Tx.alloc_shared((BM, BK), "float16")  # new shared tile
        acc = Tx.alloc_local((4,), "float32")  # per-thread accumulator
        view = Tx.decl_tensor(
            As.data, ty_args=[Tx.Tensor((BM, BK), "float16", scope="shared")]
        )  # a view over As

**A ptr-based tensor is just metadata over a pointer.** For any non-tmem tensor,
the declaration is a pointer plus a layout, and indexing resolves to an address::

    addr(tensor[coord]) = tensor.data + layout.apply(coord, shape=shape)["m"]

(``layout.apply`` returns the per-axis mapping; its ``"m"`` component is the
element offset, including the layout's origin.) Pointer arithmetic here is in
elements. Strides and origins are represented only by the layout, not by separate
tensor arguments or properties. So the *same* logical access compiles to different
address arithmetic depending purely on the tensor's metadata. Writing
``B[i, j] = A[i, j] + 1`` over a 4×8 region, with ``B`` annotated four ways in
the function signature:

.. code-block:: python

    from tvm_tirx.tirx.layout import TileLayout, S

    B: Tx.Tensor((4, 8), "float32")  # row-major
    B: Tx.Tensor((4, 8), "float32", layout=TileLayout(S[(4, 8) : (1, 4)]))  # column-major
    B: Tx.Tensor((4, 8), "float32", layout=TileLayout(S[(4, 8)] + 64))  # shifted view
    B: Tx.Tensor((4, 8), "float32", layout=TileLayout(S[(4, 8) : (16, 1)]))  # row stride 16

each makes ``B[i, j]`` lower to a different index in the generated CUDA (the
``A[i, j]`` load stays ``i*8 + j`` — only ``B``'s metadata changed):

.. code-block:: c++

    B_ptr[((i * 8) + j)]        = ...;   // row-major:        i*8 + j
    B_ptr[((j * 4) + i)]        = ...;   // column-major:     j*4 + i
    B_ptr[(((i * 8) + j) + 64)] = ...;   // layout origin 64:   i*8 + j + 64
    B_ptr[((i * 16) + j)]       = ...;   // row stride 16:    i*16 + j

Layouts must describe the actual tensor domain; lowering does not resize or
extend mismatched tile extents. For nonzero shapes and non-negative strides,
allocation footprints cover the maximum mapped memory address. A declaration
over a borrowed pointer remains responsible for the storage that pointer addresses.

Shared memory
-------------

Shared memory comes in two flavors — **static** (fixed at compile time) and
**dynamic** (sized at launch) — plus a pool helper that manages the dynamic case.

Static
~~~~~~

The simplest shared tensor is a **static** one — ``Tx.alloc_shared`` (that is,
``scope="shared"``), sized at compile time. Stage data into it, ``cta_sync`` so the
whole block sees the writes, then read it back:

.. code-block:: python

    @Tx.function
    def smem_demo(A: Tx.Tensor((128,), "float32"), B: Tx.Tensor((128,), "float32")):

        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(128,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        sm = Tx.alloc_shared((128,), "float32")  # static shared memory
        sm[tx] = A[tx]
        Tx.cuda.cta_sync()
        B[tx] = sm[tx] * Tx.float32(2.0)

It lowers to a plain ``__shared__`` array (generated CUDA, boilerplate elided):

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(128)
    smem_demo_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tx = ((int)threadIdx.x);
      __shared__ alignas(64) float sm_ptr[128];      // Tx.alloc_shared
      sm_ptr[tx] = A_ptr[tx];
      __syncthreads();                               // Tx.cuda.cta_sync()
      B_ptr[tx] = sm_ptr[tx] * 2.0f;
    }

Dynamic
~~~~~~~

**Dynamic** shared memory (``scope="shared.dyn"``) is sized per launch (the
``sharedMemBytes`` launch parameter), not at compile time. A kernel may have **only
one** dynamic-shared allocation — the *arena*. So you allocate it once and ``decl``
each tensor as a view into it: ``Tx.decl_tensor`` with the arena pointer as its
first argument and a memory-axis origin in its layout:

.. code-block:: python

    arena = Tx.alloc_tensor(
        ty_args=[Tx.Tensor((128,), "float32", scope="shared.dyn")]
    )  # the one arena
    As = Tx.decl_tensor(
        arena.data, ty_args=[Tx.Tensor((64,), "float32", scope="shared.dyn")]
    )  # offset 0
    Bs = Tx.decl_tensor(
        arena.data,
        ty_args=[Tx.Tensor(
            (64,), "float32", scope="shared.dyn", layout=Tx.TileLayout(Tx.S[(64,)] + 64)
        )],
    )  # offset 64
    As[tx] = A[tx]
    Bs[tx] = B[tx]
    Tx.cuda.cta_sync()
    C[tx] = As[tx] + Bs[tx]

Both views share the single ``extern __shared__`` arena (generated CUDA,
boilerplate elided; arena named ``smem`` for clarity):

.. code-block:: c++

    extern __shared__ __align__(64) float smem[];   // the one dynamic-shared arena
    smem[tx]      = A_ptr[tx];                       // As — view at offset 0
    smem[tx + 64] = B_ptr[tx];                       // Bs — view at offset 64
    __syncthreads();
    C_ptr[tx] = smem[tx] + smem[tx + 64];

(Two separate ``alloc_tensor`` calls with ``scope="shared.dyn"`` tensor types
are an error — *only one dynamic shared memory allocation is allowed*.) So static shared memory is sized at compile
time (``__shared__ T x[N];``); dynamic shared memory is this one launch-sized arena
with views decl'd at offsets inside it.

.. note::

   The compiler infers the required bytes from the ``shared.dyn`` allocation
   or ``SMEMPool.commit()`` and passes them as a typed launch operand.
   ``LaunchConfig.dynamic_smem_bytes`` can reserve more space; a value smaller
   than the allocation requirement raises an error. The shared launch support
   applies the same resource policy to Driver and Runtime launches, including
   repeated calls whose dynamic shared-memory requirement grows.

Pool sugar
~~~~~~~~~~

``tvm.backend.cuda.lang.SMEMPool`` automates that arena bookkeeping — it
bump-allocates the offsets so you don't ``decl`` views by hand. Beyond
``alloc`` / ``commit``, it offers
per-tensor ``align=``, an ``alloc_tcgen05_mma_AB`` helper that builds an MMA-compatible
swizzle layout for you, and ``move_base_to`` to rewind the cursor and reuse space:

.. code-block:: python

    from tvm.backend.cuda.lang import SMEMPool

    pool = SMEMPool()                             # bump allocator over shared.dyn
    As = pool.alloc((BM, BK), "float16", align=128)   # carve a tile
    Bs = pool.alloc((BK, BN), "float16", align=128)
    Cs = pool.alloc_tcgen05_mma_AB((BM, BN), "float16")  # MMA-compatible, swizzle inferred
    pool.commit()                                 # finalize the pool's size
    # pool.move_base_to(offset) rewinds the cursor to reuse space

Per-thread local storage
------------------------

Per-thread scratch uses ``scope="local"``. Allocate it with
``Tx.alloc_local(shape, dtype)``: it is private to each thread and lowers to a
local array that nvcc/ptxas can promote to registers when its accesses permit.

.. code-block:: python

    r = Tx.alloc_local((4,), "float32")   # per-thread local array
    for k in Tx.unroll(4):
        r[k] = A[tx, k]
    # ... compute on r[0..3] ...

.. code-block:: c++

    alignas(64) float r_ptr[4];          // per-thread array; ptxas may scalarize it
    r_ptr[0] = A_ptr[tx * 4 + 0];
    r_ptr[1] = A_ptr[tx * 4 + 1];
    // ...

.. note::

   The ``alignas(64)`` is the *default* tensor alignment — a tensor's
   ``data_alignment`` defaults to ``runtime::kAllocAlignment`` (64 bytes), and the
   CUDA codegen stamps it onto every allocation, including per-thread ``local``
   arrays. Statically indexed locals are typically promoted to registers by
   nvcc/ptxas (scalar replacement of aggregates), where the alignment has no
   runtime effect. Dynamically indexed arrays may instead spill to addressable
   local memory, so do not rely on register promotion when reasoning about them.

Scalar
~~~~~~

A mutable scalar is represented by a ``local`` tensor with **one element** —
strictly, you don't need a separate concept. You can allocate the tensor and
index ``[0]``:

.. code-block:: python

    phase = Tx.alloc_local((1,), "int32")   # one-element local tensor
    phase[0] = 0
    while phase[0] < 4:
        acc = acc + A[tx, phase[0]]
        phase[0] += 1

But writing ``phase[0]`` everywhere is clumsy, so a **scalar** is sugar for exactly
this — a one-element local tensor you read and write **by name**:

.. code-block:: python

    phase: Tx.int32 = 0                 # mutable scalar (sugar for the above)
    while phase < 4:
        acc = acc + A[tx, phase]
        phase += 1

    s = Tx.local_scalar("int32")        # explicit form; assign by name (s = ..., not s[0])
    acc: Tx.float32 = 0.0               # a type-annotated assignment also makes one

The two are not just similar — they parse to **structurally identical TIRx**. The
sugar is resolved entirely in the parser: ``phase: Tx.int32`` *is* that one-element
``local`` tensor, and ``phase`` / ``phase += 1`` *are* ``phase[0]`` /
``phase[0] += 1``. ``tvm.ir.assert_structural_equal`` on the two kernels passes, and
the printer even renders the explicit ``alloc_local`` + ``[0]`` form **back** as the
scalar form — so once parsing is done there is no difference at all. Both therefore
lower to the same ``alignas(64) int phase_ptr[1];``; the scalar just lets you drop
the ``[0]``. (``Tx.local_scalar`` / ``Tx.shared_scalar`` / ``Tx.alloc_scalar`` choose
the scope explicitly.)

.. note::

   **Why not a** ``Var``\ **?** A TIRx ``Var`` is *immutable* — a single static
   binding (it is exactly what ``Tx.let`` produces, below). A scalar needs to be
   *mutable* — you reassign it in loops and accumulators — so it must be backed by a
   one-element tensor you can store into repeatedly, not a ``Var``.

``let``
~~~~~~~

A ``Tx.let`` binding is **immutable** — a single TIRx ``Bind`` statement (a named
value, not a tensor). Use it for derived constants:

.. code-block:: python

    n: Tx.let = M * K               # immutable Bind
    half: Tx.let[Tx.int32] = N // 2  # ... with an explicit type

It lowers to a **plain scalar C variable** — not a tensor (no array, no ``[0]``).
For ``half: Tx.let = m * 2`` (with a runtime ``m``):

.. code-block:: c++

    int half = m * 2;     // the `let` -> a const-like local

Because the value is immutable, the simplifier is free to propagate and CSE it, so
at the use sites you often see ``m * 2`` substituted directly (or shared through a
common-subexpression temporary) rather than a reference to ``half``.

.. note::

   **Why have an immutable binding at all?** Because the value cannot change, the
   arithmetic analyzer binds the var to it (``analyzer.Bind(var, value)`` when it
   simplifies a TIRx ``Bind``), so facts proven about the value — constant bounds, the
   modular set (divisibility / alignment), ranges — **propagate through every use**.
   That feeds index simplification, bounds-check elimination, and
   alignment/vectorization decisions. A *mutable* scalar is a memory load
   (``buf[0]``): the analyzer cannot assume it stays constant, so none of those
   properties carry through. A ``let`` is also a pure value — no allocation, and
   free to inline / substitute / CSE — whereas a scalar is a one-element tensor with
   load/store semantics.

Tensor memory
-------------

Blackwell *tensor memory* is not a plain scratch scope: it must be explicitly
reserved and freed with the warp-uniform ``Tx.ptx.tcgen05.alloc`` /
``tcgen05.dealloc`` intrinsics, and each tensor is a view into it declared with
``Tx.cuda.decl_tmem(addr, ty_args=[Tx.Tensor(shape, dtype, scope="tmem", layout=layout)])``.
The ``addr`` operand is the allocated tensor-memory base address plus any desired
column offset. The declaration captures the address value when it executes, so
synchronize allocator writes to a shared address slot before declaring tensors.
Tensor-memory storage is reserved separately from the tensor declaration.
Unlike shared memory, tensor memory is not directly addressable: it is read and
written only through ``tcgen05`` ``mma`` / ``ld`` / ``st`` / ``cp``.

By hand, one warp issues the allocation into a shared slot, you ``decl`` each
tensor as a view at a column offset, and one warp frees it at the end:

.. code-block:: python

    addr = Tx.alloc_shared((1,), "uint32")             # slot for the allocated base
    if warp_id == alloc_warp:                         # tcgen05.alloc is warp-uniform
        Tx.ptx[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
            Tx.address_of(addr), Tx.uint32(512))
    Tx.cuda.cta_sync()                                # publish before capturing addr[0]
    acc = Tx.cuda.decl_tmem(
        addr[0],
        ty_args=[Tx.Tensor(
            (CTA_M, 512), "float32", scope="tmem",
            layout=tmem_layout,
        )],
    )
    # ... use acc as a gemm_async / copy_async operand ...
    if warp_id == alloc_warp:
        Tx.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()
        Tx.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
            addr[0], Tx.uint32(512))

You add any column offsets to ``addr[0]`` and manage the ``tmem_layout`` (the
matching datapath A–G layout) yourself.

See the :doc:`../../tile_ops` walkthroughs for full examples.

Tensor variable APIs
--------------------

A tensor variable is an ``ir.Var`` carrying ``tirx.TensorType`` metadata
(see *Declaring tensors* above), so most of
its methods are *compile-time* reshapes/reinterprets that change index arithmetic
or hand you a pointer — they emit no runtime op of their own.

``TensorType.__expr_methods__`` explicitly names the Python methods available
through an expression: ``B.view(...)`` binds the operand exactly as
``B.ty.view(B, ...)``. ``TensorType.__expr_properties__`` maps property names
to getters taking the type and original expression. It preserves conveniences
such as ``B.shape``, ``B.dtype``, ``B.layout``, ``B.sub`` and ``B.data`` without
modifying the shared ``Var`` class. ``B.dtype`` returns a runtime ``DataType``;
``B.ty.dtype`` stores a ``PrimType``.

Existing expression attributes and reflected fields take precedence, followed
by declared properties and then declared methods. Declared properties are
read-only. Tensor methods and properties require an ordinary tensor variable;
merely retrieving a method does not construct IR. A property getter executes
when the property is read: for example, ``B.data`` constructs the canonical
physical-pointer projection.

The common methods and properties:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - What it is
   * - ``B.data``
     - the physical-pointer projection (a ``Call``); lowers to ``B_ptr``
   * - ``B.ptr_to([i, j])``
     - a typed pointer to an element (``address_of``); prints as ``&B_ptr[…]``
   * - ``B[Tx.Ramp(i, 1, 4)]`` / ``B[Tx.Ramp(i, 1, 4)] = v``
     - a vectorized load / store; prints as ``*(float4*)(B_ptr + …)``
   * - ``B.view(*shape, layout=…)``
     - reinterpret the same storage under a new shape/layout (no copy)
   * - ``B.permute(*dims)``
     - a view with axes permuted (a transposed layout)
   * - ``T.ptr_byte_offset(B.data, byte_offset, ty=B.data.ty)``
     - a typed physical pointer offset in bytes, for passing flat addresses
       to an intrinsic

**Pointers — ``ptr_to`` / ``data``.** ``ptr_to`` is how you hand an element address
to an intrinsic or inline function; ``data`` is the base pointer:

.. code-block:: python

    B[tx] = Tx.cuda.func_call("ld", A.ptr_to([tx]), SRC, ty="float32")

.. code-block:: c++

    B_ptr[tx] = ld(&A_ptr[tx]);          // ptr_to([tx]) -> &A_ptr[tx];  A.data -> A_ptr

The pointer returned by ``ptr_to`` has the tensor's element type and storage
scope. This remains true when the tensor is a typed view over a byte-addressed
allocation pool; the pool's raw backing-pointer type does not leak through the
element address.

**Vectorized access — ``Ramp`` indices.** Move several elements as one wide
transfer (see also :doc:`data_types`):

.. code-block:: python

    B[Tx.Ramp(tx * 4, 1, 4)] = A[Tx.Ramp(tx * 4, 1, 4)]

.. code-block:: c++

    *(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);

**Reshape / reinterpret — ``view`` / ``permute``.** Both are pure metadata; the
data pointer is unchanged, only the index arithmetic differs. ``A.view(64, 4)``
sees the 256-element tensor as ``64×4``; ``A.permute(1, 0)`` transposes the axes:

.. code-block:: python

    A2 = A.view(64, 4);     y = A2[tx, 0] + A2[tx, 3]   # A2[tx, j] -> A_ptr[tx*4 + j]
    At = A.permute(1, 0);   z = At[i, j]                # At[i, j]  -> A_ptr[j*4 + i]

.. code-block:: c++

    A2_ptr[tx * 4]  /* +3 */                 // view: row-major 64x4 index
    At_ptr[(j * 4) + i]                       // permute: swapped strides

**Per-thread storage — explicit ``view``.** A thread-distributed register
layout describes both the owning thread and its local slot. To index only
the calling thread's slots, provide a memory-only layout and the physical
shape explicitly. For a tensor with eight contiguous slots per lane::

    R = Tx.alloc_tensor(
        ty_args=[
            Tx.Tensor(
                (32, 8), "float32", scope="local", layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)])
            )
        ]
    )
    R_flat = R.view(8, layout="default")   # this lane's eight physical slots
    R_2d = R.view(2, 4, layout="default")  # the same slots as a compact 2x4 tensor

The explicit layout replaces the thread distribution in the view; it does
not infer a storage shape or move data. If a register layout contains gaps,
choose a shape covering its physical span to expose those gaps, or supply
a strided memory layout to address only its occupied slots. Offsets must
also be represented explicitly. A reshape without ``layout=`` preserves
the existing logical mapping, including thread axes.
