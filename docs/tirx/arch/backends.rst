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

Backend Internals
=================

TIRx keeps the core IR and compiler machinery under ``tvm.tirx``.  A target
backend owns its script namespaces, intrinsic builders, dispatch variants,
pipeline additions, and code-generation support under ``tvm.backend``.

Loading and registration
------------------------

``tvm.backend.load`` imports a backend and calls its registration hook.  A
backend can register the following target-owned behavior:

#. TVMScript namespaces such as ``Tx.cuda``, ``Tx.ptx``, and ``Tx.nki``;
#. tile-op dispatch implementations for that target kind;
#. target tags and, when the target owns one, compilation-pipeline entry
   points; and
#. code-generation helpers that translate backend calls to target source.

Registration imports some modules only for their side effects.  Those modules
are implementation surfaces, not additional kernel-authoring APIs.

CUDA ownership
--------------

``tvm.backend.cuda`` is divided by compiler responsibility:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Responsibility
   * - ``script`` and ``ptx``
     - Construct the ``Tx.cuda``, ``Tx.ptx``, compatibility, and NVSHMEM
       namespaces.
   * - ``op``
     - Define IR builders used by the CUDA script namespaces.
   * - ``tile``
     - Register named ``Tx.cuda.tile`` instruction contracts and lowerers.
   * - ``codegen`` and ``cpp``
     - Register source-generation callbacks and CUDA C++ helpers.
   * - ``transforms``
     - Provide CUDA-specific compiler passes.
   * - ``target_tags``
     - Register named NVIDIA targets.
   * - ``lang`` and ``iket``
     - Provide reusable kernel utilities and profiling orchestration.

The PTX namespace is table-driven.  A table entry defines the legal modifier
and operand forms; construction creates a ``tirx.ptx.*`` call, and CUDA codegen
emits the corresponding instruction.  See :doc:`../api/ptx` for the supported
public forms.

Trainium ownership
------------------

``tvm.backend.trn`` follows the same boundary.  ``script`` and ``op`` construct
the ``Tx.nki`` surface, ``tensor_instructions`` registers ``Tx.trn.tile`` instruction contracts,
``layout`` and ``transform`` lower Trainium-specific memory mappings, and
``pipeline`` assembles the Trainium pass sequence.  ``target_tags`` registers
the named AWS Trainium targets.

Public integration hooks are listed in :doc:`../api/backend`; target-facing
CUDA and Trainium utilities are listed in :doc:`../api/cuda` and
:doc:`../api/trainium`.
