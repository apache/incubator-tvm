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

Tensor Instruction Registration
===============================

Backend authors define an ``Instruction`` with its canonical operator name,
ordered ``Operand`` list, reflected static attribute schema, and one lowerer.
The shared implementation is ``tvm.tirx.tile.instruction``; CUDA and
Trainium's ``tile/instructions.py`` files contain the concrete contracts.

Registration installs the fixed argument count, void return type, opaque
side effects, ``FTileOpDispatch``, printer name, and native Call validator on the Op.
The validator also runs for directly constructed Calls and printer entry.
Runtime expressions must be operands. CUDA encodes the opcode and named
instruction/descriptor parameters as normal Call arguments; only execution
scope uses a reflected attribute. See :doc:`../cuda_tile_patterns` for the
CUDA pattern extension workflow.

Lowerers receive a semantic view and ``DispatchContext`` and return a
``tvm.ir.Stmt`` directly. They can prepare pointers/descriptors and expand
layouts for the selected core instruction. They must report unsupported
contracts rather than select a different instruction family.

``FTileOpDispatch`` has the signature
``(tvm.ir.Call, DispatchContext) -> tvm.ir.Stmt``. Its presence identifies
a tile Op; no separate category registration is required. ``Instruction``
registers an adapter that decodes the semantic view and invokes its lowerer.
Custom implementations can register the callback directly using
``tvm.ir.register_op_attr(op_name, "FTileOpDispatch", dispatch)`` or C++
``OpDef::set_attr<tile::FTileOpDispatch>``. The Call must carry reflected
attributes with a required ``ThreadScope`` ``scope`` field in the target's
execution hierarchy. Direct construction may use a registered qualified name,
such as ``"cuda.thread"``, which the typed FFI boundary resolves to its enum.

Named workspace operands have fixed optional slots. Declare a workspace policy
only for instructions that own such scratch storage; do not use it to hide
caller-owned algorithms. See :doc:`../arch/tile_dispatch` for callbacks and
the statement replacement boundary.


The common Python API is under ``tvm.tirx.tile`` (also ``T.tile``):
``Instruction``, ``Operand``, ``TensorCall``, and ``DispatchContext``. Backend
implementations live under ``tvm.backend.cuda.tile`` and
``tvm.backend.trn.tile``. Their user-facing instruction paths remain
``T.cuda.tile`` and ``T.trn.tile``. Instruction operator names and attribute
schema type keys remain unchanged. Scope tags use the ``tirx.ThreadScope`` type
key; serialized data containing the former scope type must be regenerated.

Thread scopes are backend-qualified enum identities. Native backends register
their supported tags and explicit ranks with ``EnumDef``; CUDA exposes
``T.cuda.ThreadScope("warp")`` and Trainium exposes
``T.trn.ThreadScope("thread")``. A string instruction scope is resolved by the
instruction's backend before constructing its Call. Lowerers consume the enum
directly through ``DispatchContext.thread_scope``. CUDA predicate analysis,
index bindings, and active thread sets live in its private tile implementation;
Trainium does not use CUDA's execution-context model.
