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

abs / neg
=========

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst, src``.

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

``abs`` and ``neg`` operate independently on local register values. Their opcode selects
the numeric type and applicable qualifiers. Operand order is ``dst,src``; each scalar
tensor operand ends in ``(1,)`` after the group and repeat axes. An ordinary dtype view
does not change the numeric type of the requested operation.

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

Example: an explicit two-instruction scalar expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def negative_magnitude(A: T.Tensor((4, 1), "float32"), Out: T.Tensor((4, 1), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        r = T.alloc_local((4, 1), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.f32"], r, A)
        T.cuda.tile.apply(T.ptx["abs.f32"], r, r)
        T.cuda.tile.apply(T.ptx["neg.f32"], r, r)
        T.cuda.tile.apply(T.ptx["st.global.f32"], Out, r)

For finite nonzero inputs this computes ``-abs(A)``. The tile expansion has four abs
calls followed by four neg calls. This order comes from the two explicit operations, not
from a hidden unary-op combination. Later compiler optimizations may simplify the
expression. For a single caller-defined scalar expression with no exact-instruction
requirement, use :doc:`ewise`.

NaN, signed-zero, and integer boundary behavior belongs to the selected PTX type/form,
not to a generic Python ``abs`` reference. Destination storage may be gapped but must be
distinct across repeats. Register ownership cannot change between source and
destination. Predication follows :doc:`mov`.

Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``. No temporary
memory, numeric cast, barrier, or packed form is inserted.
