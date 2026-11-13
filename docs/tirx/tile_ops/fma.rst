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

fma
===

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst, a, b, c``.

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

``T.cuda.tile.apply(T.ptx["fma.rn.f32"], dst, a, b, c)`` computes the selected fused
multiply-add for each repeat. ``a*b+c`` is one native FMA atom, including its explicit
rounding qualifier. It is not a request for two independently rounded multiply/add
instructions.

Scalar tensor operands have suffix ``(1,)``. Read-only scalar operands broadcast over
repeats. Packed f16x2/bf16x2 forms use uint32 carriers; packed f32x2 uses uint64. A
packed constant must encode both values. The opcode and storage representation stay
explicit even if a contiguous layout would permit several other forms.

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

Example: four fused operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def affine_fma(A: T.Tensor((4, 1), "float32"), Out: T.Tensor((4, 1), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        r = T.alloc_local((4, 1), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.f32"], r, A)
        T.cuda.tile.apply(T.ptx["fma.rn.f32"], r, r, T.float32(2), T.float32(1))
        T.cuda.tile.apply(T.ptx["st.global.f32"], Out, r)

The arithmetic expansion has four ``fma.rn.f32`` calls. For small integer inputs the
reference is exactly ``2*A+1``. Tests intended to distinguish rounding modes must
instead use inputs whose exact product-plus-addend lies between representable outputs; a
rounded multiply followed by a rounded add is not an adequate FMA reference.

Use ``.rm``, ``.rp`` or ``.rz`` only when that form's rounding is intended and supported
by the selected type. Predication and destination preservation follow :doc:`mov`. Memory
loads/stores and synchronization remain caller code.

Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``.
