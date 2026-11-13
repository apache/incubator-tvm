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

min / max
=========

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst, a, b[, c]``.

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

``min`` and ``max`` apply the native instruction independently at every register repeat.
They do not perform a warp, block, or tensor reduction. Choose NaN, sign,
absolute-value, packed-type and other supported semantics through the instruction
spelling.

Each scalar tensor operand ends in ``(1,)``. Binary and three-input forms are
distinguished by the explicitly supplied operands, not by a layout search. ``max.f32``
with ``dst,a,b,c`` uses the registered three-input form; ``dst,a,b`` uses its binary
form. The selected target must support that form. Packed f16x2/bf16x2 variants use b32
carriers as in :doc:`add_sub_mul`.

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

Example: three-input maximum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def maximum3(
        A: T.Tensor((4, 1), "float32"), B: T.Tensor((4, 1), "float32"), Out: T.Tensor((4, 1), "float32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        a = T.alloc_local((4, 1), "float32", layout="default")
        b = T.alloc_local((4, 1), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.f32"], a, A)
        T.cuda.tile.apply(T.ptx["ld.global.f32"], b, B)
        T.cuda.tile.apply(T.ptx["max.f32"], a, a, b, T.float32(0))
        T.cuda.tile.apply(T.ptx["st.global.f32"], Out, a)

The result for finite inputs is ``maximum(maximum(A,B),0)``. The tile expansion issues
**four three-input max instructions**, not eight binary ones. NaN and signed-zero
behavior is that of the selected PTX form and must not be inferred from this
finite-input reference.

Changing ``max.f32`` to ``min.f32`` changes the operation on the same coordinate
pattern. Reducing many values still requires explicit reduction loops, shuffle
operations and/or shared communication around these atoms. In particular, a repeat axis
of length four produces four independent results; it does not accumulate into a single
output.

All inputs share group/repeat extents unless provided as scalar broadcasts. Output
locations must be distinct across repeats. Predication follows :doc:`mov`.
Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``; native form
definitions are in ``python/tvm/backend/cuda/ptx/table.py``.
