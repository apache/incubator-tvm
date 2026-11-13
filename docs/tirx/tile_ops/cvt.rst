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

cvt
===

Call parameters
---------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst, src[, src1, ...]``.

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

``cvt`` performs a numeric conversion selected by the opcode. A dtype view only
reinterprets bits; it is used here to expose the native carrier layout, not to perform
the conversion itself. Destination and source widths, operand group lengths, rounding,
saturation and packing come from the PTX form.

Scalar conversion uses tensor suffix ``(1,)`` on both sides. Packed ``cvt.rn.f16x2.f32``
has three operands: one b32 destination, a float32 source for its **high** halfword, and
a float32 source for its **low** halfword. All three have suffix ``(1,)``. The pair is
not an inferred vectorization of a scalar cast.

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

Example: eight float32 values to four packed half pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def narrow_pairs(A: T.Tensor((4, 2), "float32"), Out: T.Tensor((4, 1), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        a = T.alloc_local((4, 2), "float32", layout="default")
        d = T.alloc_local((4, 1), "uint32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], a, A)
        T.cuda.tile.apply(T.ptx["cvt.rn.f16x2.f32"], d, a[:, 1:2], a[:, 0:1])
        T.cuda.tile.apply(T.ptx["st.global.b32"], Out, d)

For each input row ``[x0,x1]``, the output word contains converted ``x0`` in its low
half and ``x1`` in its high half. Reversing the two source operands would exchange the
output elements. The expansion emits four packed cvt instructions, one per row.

.. code-block:: python

    import numpy as np
    import tvm

    x = np.linspace(-2, 2, 8, dtype="float32").reshape(4, 2)
    compiled = tvm.tirx.compile(narrow_pairs, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros((4, 1), dtype="uint32"), tvm.cuda())
    compiled(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy().view("float16"), x.astype("float16"))

The inverse operation is two scalar ``cvt.f32.f16`` instructions per packed word; there
is no corresponding ``cvt.f32x2.f16x2`` spelling in this API. Expose the packed source
with a uint16 view, add a singleton register-group axis per half, and apply the scalar
opcode. For a composite conversion expression, including explicit intermediate rounding,
use :doc:`ewise`.

Register ownership must agree on both sides even when their storage widths differ. A
wrong carrier width, missing group axis, or overlapping destination is rejected. There
is no implicit staging through shared memory.

Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``.
