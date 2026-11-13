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

ewise: caller-defined scalar compositions
=========================================

Call parameters
---------------

``T.cuda.tile.ewise(function, dst, *inputs, scope="thread")``

Operand order: ``function, dst, *inputs``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.

The scalar function defines the computation. ``scope`` is the only tile keyword.

Data representation
-------------------

function is a typed scalar TIRx Function; its parameter and return types exactly match
input and destination scalars. The atom is one scalar element, with no native
register-group suffix.

Operand specification
---------------------

``T.cuda.tile.ewise(function, dst, *inputs, scope="thread")`` applies a typed scalar
TIRx function independently to register elements. This is the one CUDA tile operation
whose atom is a caller-defined expression rather than one PTX instruction. It can
represent exp, SiLU, scale/bias, casts, or another scalar combination. Use
:doc:`add_sub_mul`, :doc:`fma`, or :doc:`cvt` when the native instruction and its
qualifiers are the contract.

The scalar atom
~~~~~~~~~~~~~~~

Tensor operands have coordinates ``(*group,*repeat)``. There is no trailing PTX
register-group axis: every coordinate after the group prefix selects one scalar
invocation. Inputs and destination have matching extents and belong to the executing
thread. Read-only zero-stride input views express tensor broadcasts; a destination must
not overlap itself.

Scalar arguments broadcast to every invocation. Their types must match the function
parameters, and its result type must match the destination dtype. The function supports
scalar bindings and returns, including an if/else with a scalar return on both paths. It
does not express reductions, memory access algorithms, or inter-thread communication.

Legality conditions
-------------------

Tensor operands are local and have identical group/repeat shapes. Scalar inputs
broadcast; tensor broadcasting is an explicit zero-stride view. Destinations belong to
the issuing thread and do not overlap. The function contains scalar bindings and a
scalar return on every path, including supported conditional returns. No
instruction-specific keyword is accepted.

Code generation
---------------

Inline the scalar function at each repeat coordinate and store its result to the
corresponding destination slot. There is one scalar function evaluation per element, not
a guaranteed single PTX instruction. Numeric conversion, packed arithmetic and
load/store staging must be written explicitly by the caller.

Examples
--------

Example: SiLU, scale, bias
~~~~~~~~~~~~~~~~~~~~~~~~~~

This complete kernel uses four scalar register values per thread. Its memory
instructions have a four-word suffix; ``ewise`` treats those same four coordinates as
scalar repeats.

.. code-block:: python

    from tvm.script import tirx as T
    from tvm.backend.cuda.axis import laneid, warpid
    from tvm.tirx.layout import S, TileLayout

    reg_layout = TileLayout(S[(2, 32, 4) : (1 @ warpid, 1 @ laneid, 1)])


    @T.function(private=True)
    def silu_affine(x: T.float32, scale: T.float32, bias: T.float32) -> T.float32:
        y = x / (T.float32(1) + T.exp(-x))
        return y * scale + bias


    @T.function
    def activation(A: T.Tensor((2, 32, 4), "float32"), Out: T.Tensor((2, 32, 4), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        R = T.alloc_local((2, 32, 4), "float32", layout=reg_layout)
        T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R, A, scope="cta")
        T.cuda.tile.ewise(silu_affine, R, R, T.float32(2), T.float32(1), scope="cta")
        T.cuda.tile.apply(T.ptx["st.global.v4.b32"], Out, R, scope="cta")

There is one vector load, four inlined scalar expressions, and one vector store per
thread. The expression computes ``silu(x)*2+1``; it does not mean ``silu(2*x+1)``. The
ordinary scalar compiler lowers the expression; ``ewise`` does not promise one PTX
instruction, choose a packed form, or fuse an unspecified activation. For an exact
exponential instruction use :doc:`ex2_lg2`.

Shared-memory load/compute/store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the destination and tensor inputs of ``ewise`` must be local registers. The
following complete kernel makes the shared staging explicit. The final read deliberately
uses another thread's row, so the second CTA barrier has a visible purpose.

.. code-block:: python

    @T.function(private=True)
    def affine(x: T.float32) -> T.float32:
        return x * T.float32(2) + T.float32(1)


    @T.function
    def shared_affine(A: T.Tensor((64, 4), "float32"), Out: T.Tensor((64, 4), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tid = T.cuda.thread_idx("x")
        S = T.alloc_shared((64, 4), "float32", layout="default", align=16)
        for j in range(4):
            S[tid, j] = A[tid, j]
        T.cuda.cta_sync()
        sv = S.view(2, 32, 4)
        R = T.alloc_local((2, 32, 4), "float32", layout=reg_layout)
        T.cuda.tile.apply(T.ptx["ld.shared.v4.b32"], R, sv, scope="cta")
        T.cuda.tile.ewise(affine, R, R, scope="cta")
        T.cuda.tile.apply(T.ptx["st.shared.v4.b32"], sv, R, scope="cta")
        T.cuda.cta_sync()
        for j in range(4):
            Out[tid, j] = S[(tid + 1) % 64, j]

The two views refer to the existing shared and local allocations. The tile operation
adds no temporary storage or barrier. If the shared store used a different ownership map
that overwrote another thread's unread input, an additional barrier would be needed
between all loads and all stores.

Compile and verify both examples:

.. code-block:: python

    import numpy as np
    import tvm

    x = np.linspace(-2, 2, 256, dtype="float32")
    for kernel, shape, ref in [
        (activation, (2, 32, 4), (x / (1 + np.exp(-x)) * 2 + 1).reshape(2, 32, 4)),
        (shared_affine, (64, 4), np.roll((x * 2 + 1).reshape(64, 4), -1, axis=0)),
    ]:
        compiled = tvm.tirx.compile(kernel, target="cuda", pipeline="tirx")
        out = tvm.runtime.tensor(np.zeros(shape, dtype="float32"), tvm.cuda())
        compiled(tvm.runtime.tensor(x.reshape(shape), tvm.cuda()), out)
        np.testing.assert_allclose(out.numpy(), ref, rtol=2e-5, atol=2e-5)

Implementation: ``python/tvm/backend/cuda/tile/patterns/ewise.py``.
