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

Thread Scope Enum Tags
======================

tvm_tirx.tirx.thread_scope
--------------------------
.. automodule:: tvm_tirx.tirx.thread_scope
   :members:
   :no-index:

Backend scopes
--------------

Thread scopes are canonical ``ffi.Enum`` identities shared by layout IR,
tile Call attributes, and dispatch contexts. Query a registered tag with
``T.ThreadScope("cuda.warp")``. The ``name`` property is the qualified enum
name; ``local_name`` is ``"warp"``. ``backend`` and ``rank`` are enum attributes,
and registration order has no hierarchy semantics.

Backends register their finite thread hierarchies natively using ``EnumDef``
and explicit backend and rank attributes. Their user-facing lookup helpers are:
``T.cuda.ThreadScope("warp")`` and ``T.trn.ThreadScope("thread")``.
``T.cuda.ThreadScope("kernel")`` names the root used by grid axes such as ``bx``
in the ``kernel -> cta`` relationship. Every scope uniformly describes a
hierarchy node; each backend tile instruction declares the scopes it supports.
Same-backend hierarchy comparisons use explicit ranks; cross-backend comparisons
are rejected.
For example, CUDA TMA instructions accept ``thread`` and ``tcgen05.ld/st`` accept
``warpgroup``. CUDA index bindings and predicate analysis live in
``tvm_tirx.backend.cuda.tile.exec_context``.

Instruction APIs continue accepting ``scope="warp"`` and also accept qualified
names or ``ThreadScope`` values. Construction resolves the value before
creating the Call; ``Call.attrs.scope`` always contains a ``ThreadScope`` enum.
Omitting scope uses the instruction backend's ``thread`` tag. Direct attribute
construction must provide the tag or its qualified name; shared attribute
schemas have no backend-independent default.

.. automodule:: tvm_tirx.backend.cuda.thread_scope
   :members:
   :no-index:

.. automodule:: tvm_tirx.backend.trn.thread_scope
   :members:
   :no-index:
