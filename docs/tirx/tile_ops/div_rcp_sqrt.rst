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

div / rcp / sqrt
================

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst, numerator, denominator; or dst, src``.

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

Select accuracy, rounding and flush-to-zero behavior in the PTX opcode. These choices
change arithmetic semantics; the tile layer does not replace a division with
multiplication by an approximate reciprocal.

.. list-table::
   :header-rows: 1

   * - Example
     - Operand order
     - Atom
   * - ``div.rn.f32``
     - ``dst, numerator, denominator``
     - One float32 per operand
   * - ``rcp.approx.ftz.f32``
     - ``dst, src``
     - One float32 per operand
   * - ``sqrt.rn.f32``
     - ``dst, src``
     - One float32 per operand

Tensor operands end in ``(1,)``, preceded by execution-group and repeat axes. Scalar
read-only inputs broadcast over those repeats. A constant numerator is allowed; its
position is not exchanged with the denominator.

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

Example: explicit division followed by square root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def root_ratio(A: T.Tensor((4, 1), "float32"), Out: T.Tensor((4, 1), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        r = T.alloc_local((4, 1), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.f32"], r, A)
        T.cuda.tile.apply(T.ptx["div.rn.f32"], r, r, T.float32(4))
        T.cuda.tile.apply(T.ptx["sqrt.rn.f32"], r, r)
        T.cuda.tile.apply(T.ptx["st.global.f32"], Out, r)

For input ``[[4],[16],[36],[64]]``, the result is ``[[1],[2],[3],[4]]``. The arithmetic
expansion contains four ``div.rn.f32`` and four ``sqrt.rn.f32`` calls. Constant folding
or instruction simplification later in the compiler may optimize them; no tile-layout
heuristic chooses a form.

To use an approximate reciprocal explicitly, replace the arithmetic with::

    T.cuda.tile.apply(T.ptx["rcp.approx.ftz.f32"], r, r)

This is a distinct numerical contract. A half-precision division composed from
conversion and scalar arithmetic belongs in :doc:`ewise` unless a registered native form
expresses exactly the requested operation.

All tensor storage is local to the executing thread. Views may select gapped registers,
but cannot move values between lanes. No memory staging or barrier is inserted.
Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``.
