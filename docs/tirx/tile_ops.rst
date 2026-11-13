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

Tile Operations
===============

CUDA tile operations apply an explicit PTX instruction or a scalar function to
canonical tensor views::

    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], registers, global_words)
    T.cuda.tile.apply(T.ptx["st.shared.v4.b32"], shared_words, registers)

See :doc:`cuda_tile_patterns` for every operand pattern, descriptor contract,
execution-group axis, and caller view. ``T.cuda.tile.ewise`` applies a
caller-defined scalar TIRx function to registers. No CUDA tile call selects a
vector width, matrix shape, transpose mode, or staging algorithm from layouts.

Both backends use ordinary opaque void Calls. Tensor regions, pointers,
predicates, coordinates and descriptor parameters are normal IR arguments.
The shared ``FTileOpDispatch`` boundary remains unchanged. ``T.ptx`` and
``T.nki`` expose raw scalar/address instructions.

.. toctree::
   :hidden:

   cuda_tile_patterns

Trainium instructions
---------------------

``T.trn.tile`` exposes ``load``, ``store``, ``tensor_copy``, ``matmul``,
``activation``, ``reciprocal``, ``memset``, ``tensortensor``, ``tensorscalar``,
``tensorreduce``, and ``affine_select``. The selected binary instruction must
match the operand layout: broadcasting along a free dimension can require
``tensorscalar`` even when the scalar is supplied by another tensor.

``scalar_tensor_scalar``, ``scalar_tensor_tensor``, ``tensorscalar_reduce``,
and ``activation_reduce`` represent native fused instructions. Static
``opcode``, ``op0``, ``op1``, ``reduce_op``, and ``axes`` qualifiers specify
their arithmetic. Scale, bias, and all tensor operands remain in ``args``.

Named optional ``acc_psum``, ``const_bias``, and ``partial_reduce`` operands
allow caller-provided workspace. The private allocation pass still allocates
and reuses omitted workspace for the instructions that need it. In IR, an
absent operand occupies its fixed slot as an empty ``Tuple``; TVMScript prints
it as ``None``.

Algorithms owned by the caller
------------------------------

CUDA reductions, layout permutation, synchronous global/shared copying, and
scalar fallback copying are implemented directly by the caller.
Write the loops, temporary registers, and synchronization at the call site.
For an in-place shared-memory permutation, finish all reads into registers,
synchronize the warp, write using the destination layout, and synchronize
again before reusing the storage.

Trainium ``tensor_copy`` rejects partition-axis transposition. A caller using
matrix multiplication for transposition allocates and initializes an identity
tensor, allocates PSUM, calls ``matmul(..., transpose_A=True)``, and optionally
calls ``tensor_copy`` to move the result back to SBUF. Copy lowering does not
allocate an identity or select a matrix multiplication algorithm.

See :doc:`api/tile` for the constructor signatures and
:doc:`arch/tile_dispatch` for the lowering boundary.

CUDA instruction guides
-----------------------

Pages follow PTX instruction families. Each page specifies the operand image,
an explicit caller view, and the resulting instruction count. Asynchronous
completion belongs to the selected instruction family, not to a generic
``copy_async`` or ``gemm_async`` operation.

.. toctree::
   :maxdepth: 1

   tile_ops/ld_st
   tile_ops/ldmatrix_stmatrix
   tile_ops/mov
   tile_ops/add_sub_mul
   tile_ops/div_rcp_sqrt
   tile_ops/min_max
   tile_ops/fma
   tile_ops/cvt
   tile_ops/ex2_lg2
   tile_ops/abs_neg
   tile_ops/cp_async
   tile_ops/cp_async_bulk
   tile_ops/cp_async_bulk_tensor
   tile_ops/tcgen05_ld_st
   tile_ops/tcgen05_cp
   tile_ops/mma_sync
   tile_ops/tcgen05_mma
   tile_ops/ewise

Trainium reductions
-------------------

.. toctree::
   :maxdepth: 1

   tile_ops/reduction
