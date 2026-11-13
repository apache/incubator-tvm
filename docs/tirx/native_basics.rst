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

CUDA Programming Guide
======================

.. note::

   Native-level kernel authoring for the **CUDA backend** (the ``"cuda"``
   target): the thread hierarchy, memory scopes, the ``Tx.cuda.*`` / ``Tx.ptx.*``
   intrinsics, and the compile / run / inspect loop. The chapters build from a
   complete ``scale`` example through shared-memory and warp-level kernels.

What "native level" means
-------------------------

A native-level TIRx kernel reads like a structured device kernel: you place
threads yourself, allocate shared/per-thread local tensors, write loops and barriers, and
call device intrinsics directly. You explicitly choose the orchestration and
layouts; standard lowering still dispatches primitives, applies layouts, and
vectorizes or unrolls marked loops. This is the foundation the tile ops
(:doc:`tile_ops`) are built on; everything here is what those ops
ultimately lower to, so it is also where you go when a hardware feature does not
have a primitive yet.

The authoring model
-------------------

- ``@Tx.function`` (or ``@Tx.jit`` for compile-time-specialized) kernels, written
  with ``from tvm.script import tirx as Tx``;
- ``Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(32,)))`` plus *scope-id* intrinsics for thread binding;
- ``Tx.Tensor`` parameter annotations and ``Tx.alloc_*`` scratch tensors;
- ordinary loops, branches, and scalar math;
- ``tvm.compile(mod, target=..., tir_pipeline="tirx")`` to build, then call the
  result directly.

All native authoring uses these imports. The ``__future__`` import lets ``@Tx.jit``
kernels reference compile-time parameters inside type annotations (see
:doc:`native_basics/cuda/functions`); it is harmless for ordinary kernels::

    from __future__ import annotations
    import tvm
    from tvm.script import tirx as Tx

Start here
----------

If this is your first TIRx kernel, complete
:doc:`First Kernel <native_basics/cuda/first_kernel>` before using the chapters
below as a language reference.

Language guide
--------------

.. toctree::
   :maxdepth: 1

   native_basics/cuda/functions
   native_basics/cuda/data_types
   native_basics/cuda/tensors
   native_basics/cuda/control_flow

CUDA execution and compilation
------------------------------

.. toctree::
   :maxdepth: 1

   native_basics/cuda/threads_sync
   native_basics/cuda/compiling

Advanced topics
---------------

.. toctree::
   :maxdepth: 1

   native_basics/cuda/parser_utils
   native_basics/cuda/profiling
