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

add / sub / mul
===============

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst, lhs, rhs``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`../cuda_tile_patterns`. It does not elect an issuing thread.
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

Each operand uses the scalar PTX register type or its equal-width bit container. A
packed f16x2/bf16x2 value occupies one b32 container; f32x2 occupies one b64 container.
The final tensor axis counts native registers, not the packed scalar elements. Rounding,
saturation, signedness and packing are selected in the opcode.

Operand specification
---------------------

The opcode chooses the numeric type, rounding, saturation and packing mode. The operand
order is ``dst, lhs, rhs``. For a scalar instruction, each tensor ends in a singleton
register-group axis; for example ``(8,1)`` at thread scope means eight scalar
instructions. A read-only scalar expression may replace an input tensor and broadcasts
over all eight repeats.

Native packed arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~

Packed arithmetic still has one register **carrier** per operand:

.. list-table::
   :header-rows: 1

   * - Example opcode
     - Carrier dtype
     - Values per carrier
   * - ``add.rn.f32``
     - float32
     - One float32
   * - ``add.rn.f16x2``
     - uint32 (raw bits)
     - Two float16
   * - ``mul.rn.bf16x2``
     - uint32 (raw bits)
     - Two bfloat16
   * - ``add.rn.f32x2``
     - uint64 (raw bits)
     - Two float32

Not every qualifier is valid for every type or architecture. PTX form validation checks
the spelling and the compiler checks the target. Packing is a caller decision, not the
consequence of an even tensor extent.

Legality conditions
-------------------

All tensor operands are local registers owned by the issuing thread. Their group/repeat
shapes agree; the final extent equals that operand's native register count (one for a
scalar operand). Scalar expressions are allowed for single-register read inputs and
broadcast across repeats. Destination slots must not overlap. Register storage strides
are in container elements; they need not be one. The selected PTX form validates operand
types.

Code generation
---------------

For each group and row-major repeat coordinate, bind each tensor's final register axis
in increasing order and emit one selected instruction. A scalar input is reused. No
packing, memory staging or synchronization is inserted. The number of expanded
instructions per issuing thread is the product of repeat extents; downstream compiler
optimizations may simplify unused or constant results.

Examples
--------

Example: eight packed half additions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel accepts packed input words and leaves the result packed:

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def packed_add(
        A: T.Tensor((8, 1), "uint32"), B: T.Tensor((8, 1), "uint32"), Out: T.Tensor((8, 1), "uint32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        a = T.alloc_local((8, 1), "uint32", layout="default")
        b = T.alloc_local((8, 1), "uint32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.b32"], a, A)
        T.cuda.tile.apply(T.ptx["ld.global.b32"], b, B)
        T.cuda.tile.apply(T.ptx["add.rn.f16x2"], a, a, b)
        T.cuda.tile.apply(T.ptx["st.global.b32"], Out, a)

.. code-block:: python

    import numpy as np
    import tvm

    x = np.arange(16, dtype="float16").reshape(8, 2)
    y = np.full((8, 2), 2, dtype="float16")
    compiled = tvm.tirx.compile(packed_add, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros((8, 1), dtype="uint32"), tvm.cuda())
    compiled(
        tvm.runtime.tensor(x.view("uint32"), tvm.cuda()),
        tvm.runtime.tensor(y.view("uint32"), tvm.cuda()),
        out,
    )
    np.testing.assert_array_equal(out.numpy().view("float16"), x + y)

The arithmetic expansion is eight ``add.rn.f16x2`` instructions, processing sixteen half
values. It does not insert a pack/unpack conversion. A scalar constant passed to a
packed opcode must already represent the whole packed carrier; a numeric uint32 constant
``2`` is not the encoding of two half 2s.

Subtraction and multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sub`` preserves the written operand order, including a scalar on the left. ``mul``
uses the selected floating-point form or the integer ``lo``/``hi``/ ``wide`` form and
its declared result type. Different result widths require the matching destination
dtype, not a silent reinterpretation.

Tensor operands share repeat extents and must have the correct thread owner. An input
broadcast is a scalar or an explicit zero-stride view. Output storage cannot overlap
across repeats. ``pred`` and ``preserve_dst`` follow the register instruction contract
described in :doc:`mov`.

For exp/SiLU plus scale/bias, define the scalar expression in :doc:`ewise`.
Shared-memory inputs require explicit loads and stores around the arithmetic.
Implementation: ``patterns/register.py`` under ``python/tvm/backend/cuda/tile``.
