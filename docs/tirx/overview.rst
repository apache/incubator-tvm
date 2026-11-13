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

Overview
========

TIRx (pronounced *"tier-ex"*) is an open-source, hardware-native DSL and compiler
for machine-learning kernels. It targets the part of the AI software stack
where fast-moving kernels meet fast-moving hardware: TIRx compiles to GPUs and
specialized AI accelerators today and is designed to grow with the hardware
generations that follow. The same design makes it a substrate not only for
expert-written kernels, but also for agent-generated kernels and megakernel
systems.

TIRx is the next-generation kernel-level compiler structure of Apache TVM, and
is built on top of the TVM compiler infrastructure.

.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/tirx/tirx_overview_hero.png
   :align: center
   :alt: A TIRx kernel keeps orchestration in hardware-native source while exposing tile
         structure to the compiler; agentic, megakernel, and new-backend systems
         build on the same structure.

   A TIRx kernel keeps orchestration — pipeline state, roles, synchronization,
   and backend intrinsics — in hardware-native source, while execution scope,
   tensor layout, and tile op dispatch expose the recurring tile
   structure to the compiler. Higher-level systems build on the same structure.

Design Philosophy
-----------------

Kernel DSLs are most effective when they choose the right boundary between the
programmer and the machine. For mature kernels on mature hardware, that boundary
can be high-level: the compiler hides thread assignment, memory movement, layout
details, and instruction selection behind compact tensor or tile abstractions,
and this works well for established kernel patterns.

At the frontier, the same boundary is under more pressure. New instructions,
memory spaces, cooperation patterns, and kernel algorithms often appear *before*
a compiler has enough built-in machinery to automate them well.

TIRx chooses a **lower and more explicit boundary**. It keeps the parts of a
kernel that frequently require expert control — pipeline structure,
synchronization, role assignment, memory placement, and backend intrinsics — in
hardware-native source code. At the same time, it exposes the recurring tile-level
structure to the compiler through three constructs: *execution scope*, *tensor
layout*, and *tile op dispatch*. Orchestration stays in hardware-native
source code, while the recurring tile-level structure becomes visible to the
compiler. Hardware-native control is powerful but costs engineering effort;
exposing recurring operations as tile ops relieves this, since authors
reuse a dispatched implementation instead of re-writing the same operation for
each kernel and backend.

The result is a DSL that can **grow with the hardware**. A new feature can first
be used directly as a native intrinsic, and later become a reusable tile op
once the pattern stabilizes. This is the core design philosophy behind TIRx:
keep the foundation small and explicit, and let the backend library evolve as
new accelerator generations arrive.

.. _tirx-programming-model:

The Programming Model
---------------------

A TIRx program reads as a structured native kernel: loops, branches, tensors,
synchronization, pipeline state, backend intrinsics, and hardware roles are
written directly. Tile ops appear exactly where a repeated hardware-level
operation should become reusable and dispatchable.

Authoring layers and IR
~~~~~~~~~~~~~~~~~~~~~~~

TIRx programs use one TVMScript dialect alias::

   from tvm.script import tirx as Tx

The namespace exposes three authoring layers.  They share the same TIRx
program, but enter the compiler at different levels:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Authoring layer
     - Syntax
     - Representation and lowering
   * - Core language
     - ``Tx.*``
     - Creates TIRx statements, expressions, tensors, and control flow.
   * - Tile ops
     - ``Tx.cuda.tile.*`` / ``Tx.trn.tile.*``
     - Creates an ordinary void ``Call`` that is replaced by its backend instruction
       implementation during tile dispatch.
   * - Backend operations
     - ``Tx.cuda.*`` and ``Tx.ptx.*``
     - Creates backend calls directly; these calls bypass tile dispatch.

``tvm.script.tirx`` is the construction syntax for the ``tvm.tirx`` object
model, not a separate IR.  Tile dispatch and layout/scope lowering replace
high-level TIRx nodes with lower-level TIRx statements and calls.  They do not
convert the program to the separate ``tvm.tir`` object model.

.. code-block:: text

   Tx.* ──────────────────────────────▶ TIRx statements and expressions ──────┐
   backend.tile.* ─▶ Evaluate(Call) ─▶ instruction lowering ─────────────────────────┤
   Tx.cuda.* / Tx.ptx.* ─────────────▶ backend calls ─────────────────────────┤
                                                                               ▼
                                             layout and scope lowering ─▶ codegen

The model has three core ingredients.

Execution scope
~~~~~~~~~~~~~~~~~

Execution scope describes both the active participants and the logical scope of
a tile op invocation. Control flow such as ``if wg_id == ...``,
``warp_id == ...``, or ``cbx == ...`` selects which hardware roles enter a
region, while predicates such as ``Tx.cuda.elect_sync()`` further select the
issuing lane within each active warp. To obtain one issuer for a wider scope,
first select one warp and then use ``elect_sync`` inside it.

Tile ops declare their execution scope explicitly. For example,
``scope="warpgroup"`` selects warpgroup cooperation; ``scope="thread"``
is the default on backend tensor instructions.

Tensor layout
~~~~~~~~~~~~~~

Tensor layout, with a storage-first interface, describes how logical
tensors map to physical resources. A tile may live in global memory, shared
memory, per-thread local storage, tensor memory, or accelerator SRAM.
Users declare where each tile lives and how its elements are spread across
lanes, warps, and per-thread storage elements; tile op lowering uses
those declarations to validate operand mappings and generate addresses. A layout
is a storage description, not a loop-transformation utility: users may construct
a tile's layout, but never use layouts to transform loops.

.. seealso::

   The layout model — including the shard / replica / offset structure and its
   design rationale — is described in :doc:`layout`, which also has an
   interactive explorer.

Tile op dispatch
~~~~~~~~~~~~~~~~

Each backend registers lowering implementations for its tile ops. On CUDA,
the caller selects an exact PTX instruction, including its qualifiers::

   Tx.cuda.tile.apply(Tx.ptx["ld.global.v4.b32"], registers, global_words)

Each instruction defines a canonical operand pattern. The caller prepares
views with the required atom coordinates and repeat axes; lowering checks the
operand shapes, physical layouts, and execution scope, then emits that same
instruction for each repeat coordinate. The caller controls vector width,
matrix shape, transpose mode, temporary storage, and synchronization.
``Tx.cuda.tile.ewise`` repeats a caller-defined scalar function over registers.

Trainium exposes named instruction ops such as ``load``, ``store``,
``tensor_copy``, ``matmul``, and ``tensorreduce``. Their lowering checks the
instruction's operand and layout requirements and generates the corresponding
NKI calls.

Putting it together
~~~~~~~~~~~~~~~~~~~~~

Across a full kernel, orchestration stays in ordinary source code, and the
recurring hardware operations appear as tile ops. For the building blocks —
TMA loads, ``tcgen05`` async MMA, Trainium reductions, and the rest — see the
instruction walkthroughs in :doc:`tile_ops`.

What TIRx Enables
-----------------

TIRx is immediately useful as a kernel DSL. The same structure also helps with
three things that are becoming important for ML systems: supporting new
hardware, building megakernels, and agentic kernel programming.

A stable extension boundary for future hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New hardware support is a staged process rather than a redesign of the DSL: a
feature is first exposed as a backend intrinsic, then promoted into a tile
op once the usage pattern repeats. Future hardware grows the backend
library, not the core language.

Megakernels and composable tile tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because TIRx tasks exist as compiler IR rather than separately compiled kernels,
a megakernel compiler can stitch and schedule them directly — re-offsetting
shared memory, renaming barriers, reassigning warp roles, and interleaving
pipelines across tasks.

.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/tirx/tirx_megakernel.png
   :align: center
   :alt: Kernel-by-kernel and CUDA-graph execution leave kernel boundaries
         between tasks, while a megakernel fuses tasks into one kernel with
         fine-grained dependencies and in-kernel scheduling.

   From kernel-by-kernel launches to a single megakernel: fusing tasks into one
   kernel exposes fine-grained dependencies and in-kernel (static or dynamic)
   scheduling.

Agentic kernel programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TIRx exposes its IR and compiler utilities through TVM FFI across Python, C++,
and Rust. Its structured IR supports inspection and well-formedness checks
before a kernel is benchmarked, while target compilation and execution provide
the final validation of backend-specific behavior.

.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/tirx/tirx_agentic.webp
   :align: center
   :alt: Agent-visible compiler infrastructure exposes TIRx IR and utilities via
         TVM FFI across Python, Rust, and C++, over a structured search space
         spanning levels L1 to L4.

   Agent-visible compiler infrastructure (IR and utilities over TVM FFI) plus a
   structured search space (levels L1–L4) make TIRx a compiler-mediated
   optimization surface for agents.

Next Steps
----------

- :doc:`install` — install TIRx and the kernel library.
- :doc:`native_basics/cuda/first_kernel` — write and compile a first CUDA kernel.
- :doc:`programming/index` — the complete programming guide.
- :doc:`layout` and :doc:`tile_ops` — reusable tensor and tile abstractions.
- :doc:`arch/index` — compiler internals (lowering pipeline, passes, codegen).
- :doc:`api/index` — authoring, IR, compiler, and extension API signatures.
