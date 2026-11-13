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

ex2 / lg2
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

These PTX instructions calculate base-two exponential and logarithm. They are not
natural exp/log. Select the explicit approximation and subnormal policy, for example
``ex2.approx.ftz.f32`` or ``lg2.approx.ftz.f32``. The tile layer forwards this spelling
without selecting a replacement.

Both tensor operands use local registers and end in ``(1,)``; each preceding repeat
selects another scalar input/output. Output repeats cannot alias. A scalar read-only
input broadcasts over all repeats.

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

Example: four base-two exponentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def exp2_native(A: T.Tensor((4, 1), "float32"), Out: T.Tensor((4, 1), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        r = T.alloc_local((4, 1), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.f32"], r, A)
        T.cuda.tile.apply(T.ptx["ex2.approx.ftz.f32"], r, r)
        T.cuda.tile.apply(T.ptx["st.global.f32"], Out, r)

Inputs ``[-1,0,1,2]`` give the power-of-two reference ``[0.5,1,2,4]``. The arithmetic
expansion contains four ex2 instructions. For general inputs, test with tolerances
appropriate to the approximate opcode; ``.ftz`` is part of the chosen handling of
subnormal values.

Use ``lg2.approx.ftz.f32`` with the same operand pattern for base-two logarithms. If the
operation is natural exp, SiLU, or an exp/scale/bias combination, write that scalar
expression through :doc:`ewise`. The scalar compiler then owns its expansion; the call
is not documented as one ex2.

Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``.
