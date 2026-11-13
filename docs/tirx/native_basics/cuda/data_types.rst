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

Data types and expressions
==========================

Every TIRx expression carries a low-level **dtype** and a high-level **type**.

Expression dtypes
-----------------

A ``PrimExpr``'s ``.dtype`` is its scalar (or vector) element type — ``float32``,
``float16``, ``bfloat16``, ``int32``, ``uint8``, ``bool``, the low-precision
``float8_e4m3fn`` / ``float4_e2m1fn`` …, ``handle`` (a pointer), and vector forms
such as ``float32x4``. Each prints to the matching CUDA type. Allocating local and
shared tensors across several dtypes, plus a vectorized ``float32x4`` load/store:

.. code-block:: python

    @Tx.function
    def dtypes(A: Tx.Tensor((256,), "float32"), O: Tx.Tensor((256,), "float32")):

        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(64,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        f16 = Tx.alloc_local((1,), "float16")  # per-thread locals ...
        bf16 = Tx.alloc_local((1,), "bfloat16")
        i32 = Tx.alloc_local((1,), "int32")
        u8 = Tx.alloc_local((1,), "uint8")
        b1 = Tx.alloc_local((1,), "bool")
        sm = Tx.alloc_shared((64,), "float16")  # ... and a shared tile
        v = Tx.alloc_local((1,), "float32x4")  # a vector-dtype local (float4)
        v[0] = A[Tx.Ramp(tx * 4, 1, 4)]  # vectorized load
        O[Tx.Ramp(tx * 4, 1, 4)] = v[0]  # vectorized store
        # ... (use f16/bf16/i32/u8/b1/sm) ...

lowers to (generated CUDA, elided):

.. code-block:: c++

    half          f16_ptr[1];               // float16
    nv_bfloat16   bf16_ptr[1];              // bfloat16
    int           i32_ptr[1];               // int32
    uchar         u8_ptr[1];                // uint8
    bool          b1_ptr[1];                // bool
    __shared__ alignas(64) half sm_ptr[64]; // shared float16
    float4        v_ptr[1];                 // float32x4  (vector)
    v_ptr[0]                  = *(float4*)(A_ptr + tx * 4);   // vectorized load
    *(float4*)(O_ptr + tx * 4) = v_ptr[0];                   // vectorized store

A tensor's dtype can itself be a **vector type**: ``Tx.alloc_local((1,), "float32x4")``
declares a per-thread ``float4`` value (you index it as ``v[0]``), and a
Indexing a scalar tensor with a four-lane ``Ramp`` moves the vector as one
16-byte access. Any tensor or scalar can carry a vector dtype.

The dtype → CUDA mapping is:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - dtype → CUDA
     - dtype → CUDA
     - dtype → CUDA
   * - ``float32`` → ``float``
     - ``float16`` → ``half``
     - ``bfloat16`` → ``nv_bfloat16``
   * - ``int32`` → ``int``
     - ``uint8`` → ``uchar``
     - ``bool`` → ``bool``
   * - ``float32x4`` → ``float4``
     - ``handle`` → ``T*`` (pointer)
     - (vector dtypes → CUDA vector types)

dtype vs type
-------------

The ``dtype`` is *low-level* — it says "what bits". Separately, a value has a
high-level **type**: ``PrimType(dtype)`` for a scalar, or
``PtrType(PrimType(dtype), scope)`` for a pointer. Most expressions are scalars
(``PrimType``); the type system matters mainly for **pointers**.

Pointers (``handle``)
---------------------

A tensor's ``data`` — its pointer — is a ``Var`` of pointer type, and it is
**immutable** (a pointer is never reassigned). That shapes how you obtain one:

- ``Tx.alloc_tensor(...)`` allocates storage **and** defines its ``data`` pointer.
- ``Tx.decl_tensor(ptr, ty_args=[Tx.Tensor(...)])`` declares a tensor over an existing pointer
  ``Var`` ``ptr``.
- To back a tensor with a pointer **expression** — e.g. ``Tx.ptx.mapa`` giving
  another cluster CTA's shared address — convert the ``uint64`` address the
  instruction wrote to a pointer with the intended element type and storage
  scope.  Assigning that pointer expression to an unannotated name
  automatically creates an immutable ``Bind`` (and ``data`` receives the
  resulting pointer ``Var``):

  .. code-block:: python

      from tvm.ir.type import PtrType, PrimType

      mapped = Tx.alloc_local([1], "uint64")
      Tx.ptx.mapa.u64(mapped[0], mbar.ptr_to([0]), Tx.uint32(0))
      ptr_ty = PtrType(PrimType("uint64"), "shared")
      ptr = Tx.reinterpret(mapped[0], ty=ptr_ty)
      remote_mbar = Tx.decl_tensor(ptr, ty_args=[Tx.Tensor([1], "uint64", scope="shared")])

  Pointer bindings cannot be reassigned; use a new name for a different
  pointer value.
