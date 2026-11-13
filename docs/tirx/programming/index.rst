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

Programming Guide
=================

This guide explains how to write, compile, and inspect TIRx kernels.  Start
with the CUDA guide for the core language and native programming model, then
use tensor layouts and tile ops to express reusable tile-level work.

The examples use a single alias for the TIRx TVMScript dialect::

   from tvm.script import tirx as Tx

Core language constructs are written as ``Tx.*``.  Backend-specific helpers
and direct CUDA instructions are available through ``Tx.cuda.*`` and
``Tx.ptx.*``.  Reusable tile operations are written as ``Tx.cuda.tile.*`` / ``Tx.trn.tile.*``.

.. toctree::
   :maxdepth: 1

   ../native_basics
   ../layout
   ../tile_ops

For callable reference entries, see the :doc:`API reference <../api/index>`.
For lowering, dispatch, and backend implementation details, see
:doc:`Compiler Internals <../arch/index>`.
