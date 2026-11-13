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

Core TVMScript
==============

TIRx kernels use ``tvm_tirx.script.tirx`` for the parser and core IR builders::

   from tvm_tirx.script import tirx as Tx

   Tx.alloc_tensor(...)

Tile ops and backend-specific namespaces are documented separately in
:doc:`tile`, :doc:`cuda`, and :doc:`ptx`.  For the relationship between these
authoring layers and TIRx IR, see :ref:`tirx-programming-model`.

Parser entry points
-------------------
.. automodule:: tvm_tirx.script.tirx
   :members: function, jit, inline, macro
   :undoc-members:
   :no-index:

Core IR builder
---------------

Common statement frames and construction helpers live in
``tvm_tirx.script.ir_builder.frame`` and ``tvm_tirx.script.ir_builder.stmt``. The TIRx
namespace re-exports them alongside its function, tensor/layout, and execution
scope builders. Native shared statement builders live under
``include/tvm/script/ir_builder`` and ``src/script/ir_builder``; dialect-specific
thread-placement and tensor alias policies remain in the TIRx/S-TIR builders.

.. automodule:: tvm_tirx.tirx.script.ir_builder.ir
   :members:
   :no-index:
   :exclude-members: LetAnnotation

.. currentmodule:: tvm_tirx.tirx.script.ir_builder.ir

.. py:class:: LetAnnotation(type_spec=None)

   Marker used by ``Tx.let`` and ``Tx.let[dtype]`` annotations to construct an
   explicit ``LetStmt``.

Tracing ordinary Python
-----------------------

The native builders also support frontends that execute ordinary Python to
construct IR. ``IRBuilder.line_trace_scope()`` records file-backed Python source
lines. Script-facing void calls automatically emit into an active IRBuilder,
with or without line tracing::

   from tvm_tirx.script import tirx as Tx
   from tvm_tirx.script.ir_builder import IRBuilder
   from tvm_tirx.tirx.script import ir_builder as I

   with IRBuilder() as builder:
       with I.function_():
           I.func_name_("synchronize")
           with builder.line_trace_scope():
               with Tx.device_entry(
                   launch=Tx.cuda.LaunchConfig(grid=1, block=32)
               ):
                   Tx.ptx["bar.sync"](0)
   function = builder.get()

Auto emission applies to script-facing calls through ``Tx``, including PTX
and backend namespaces. A void call returns an ``AlreadyEmitted`` receipt;
passing that receipt to ``evaluate`` or ``emit_`` does not emit it again.
Calls producing values remain expressions. Raw IR constructors such as
``tvm_tirx.ir.Call``, ``tvm_tirx.tirx.op`` and raw backend APIs remain pure even inside
an IRBuilder. Line tracing only supplies source locations; it neither enables
nor disables emission. The trace scope restores its previous trace and location
state on exit, including exceptional exits, and applies only to its owning builder.
Explicit ``with_loc`` and ``at_`` locations compose with the traced lines.

Line tracing excludes TVM and standard-library implementation files. Frontends
can supply additional directories using ``line_trace_scope(exclude_paths=(... ,))``.
User helpers retain their own source line and the caller's location through
``CallSiteLoc``. These locations cover whole lines; the TVMScript parser records
expression ranges. Python loops and branches execute during construction; use
the native statement frames to construct runtime control flow.

Tensor parameters and shared dimensions
---------------------------------------

Traced frontends can bind native types directly with ``I.arg_``. A shared symbolic
dimension imposes the same size constraint on multiple tensor arguments::

   n = Tx.Var("n", Tx.int64)
   # Inside an I.function_() frame:
   x = I.arg_("x", Tx.Tensor((n,), Tx.float32))
   y = I.arg_("y", Tx.Tensor((n,), Tx.float32))

The compiled host entry accepts tensor arguments, including PyTorch tensors,
and checks their dtype, rank and shared dimensions. Use independent symbolic
dimensions for unrelated sizes. If the same dimension is also an explicit scalar
parameter, ``I.arg_("n", n)`` binds that existing Var. ``x.data`` and
``x.ptr_to(indices)`` provide pointers inside the function. ``Tx.Ptr`` retains
its raw-pointer ABI and does not imply tensor argument conversion.

Explicit register fragments
---------------------------

A register fragment is an ordinary local tensor with an explicit layout.
For a two-CTA Layout-B accumulator, define the TMEM and register layouts
outside the kernel::

   from tvm_tirx.backend.cuda.axis import tid_in_wg
   from tvm_tirx.backend.cuda.layout import tmem_mma_operand_layout
   from tvm_tirx.tirx.layout import TileLayout, S

   layout_b = tmem_mma_operand_layout(
       "D", (64, 128), "float32", M=128, cta_group=2)
   register_layout = TileLayout(S[(64, 2, 64) : (1 @ tid_in_wg, 64 @ tid_in_wg, 1)])

Inside the kernel, declare the accumulator over its allocated TMEM address,
allocate the register fragment, and prepare matching instruction views::

   C = Tx.cuda.decl_tmem(tmem_address, ty_args=[
       Tx.Tensor((64, 128), "float32", scope="tmem", layout=layout_b)
   ])
   frag = Tx.alloc_tensor(ty_args=[
       Tx.Tensor((64, 128), "float32", scope="local", layout=register_layout)
   ])
   C_words = C.rearrange("(wh r) (wl c) -> (wl wh) r c",
                         wh=2, r=32, wl=2)
   frag_words = frag.rearrange("(wh r) (wl c) -> (wl wh) r c",
                               wh=2, r=32, wl=2)
   Tx.cuda.tile.apply(
       Tx.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"],
       frag_words, C_words, scope="warpgroup")
   Tx.ptx.tcgen05.wait__ld.sync.aligned()

``tmem_address`` must name a live allocation with at least 64 columns.
The readback requires a full active warpgroup.
Complete the producing MMA's commit/mbarrier wait and
the required ``tcgen05.fence::after_thread_sync`` before the load.

Layout B stores logical ``C[i,j]`` in TMEM lane
``i + 64*(j//64)`` and column ``j%64``. The register layout assigns that
same element to thread ``i + 64*(j//64)`` and local slot ``j%64``.
The views above both have shape ``(4,32,64)``: ``warp=2*wl+wh`` chooses
the physical 32-lane slab, ``r`` selects its row, and ``c`` selects its
64 columns. Each warp issues one ``.32x32b.x64`` instruction and each
lane receives 64 registers. No temporary copy is needed.
``wait::ld`` makes the loaded registers available to subsequent
instructions; it does not wait for an earlier MMA.

See :doc:`../tile_ops/tcgen05_ld_st` for other native shapes,
pack/unpack, and a complete store/load example.
