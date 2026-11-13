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

.. _tensor-ir-deep-dive:

TensorIR
========
TensorIR is one of the core abstractions in the Apache TVM stack, used to
represent and optimize primitive tensor functions.

The TensorIR codebase consists of two modules (split from the former ``tir``):

- **tirx** — Core IR definitions and lowering (Function, Buffer, SBlock,
  expressions, statements, lowering passes).
- **s_tir** (Schedulable TIR) — Schedule primitives, MetaSchedule, DLight,
  and tensor intrinsics.

In TVMScript, both modules are accessed via
``from tvm.script import tirx as T``.

.. toctree::
    :maxdepth: 2

    abstraction
    learning
    tutorials/tir_creation
    tutorials/tir_transformation
    tutorials/dlight_gpu_scheduling
    tutorials/meta_schedule
