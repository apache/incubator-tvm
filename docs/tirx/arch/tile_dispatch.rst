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

Tensor Instruction Lowering
===========================

Backend tensor operations are ordinary ``tvm.ir.Call`` nodes with opaque
side effects and void return types. CUDA and Trainium register their contracts
in ``python/tvm/backend/{cuda,trn}/tile/instructions.py`` using the shared
``python/tvm/tirx/tile/instruction.py`` machinery. C++ reflected attribute
types and the native Call validator are in ``src/tirx/tile/op.cc``.

The public C++ tile interfaces are grouped under ``include/tvm/tirx/tile/``:

- ``op.h`` declares the instruction registration and validation contract.
- ``dispatch.h`` declares ``FTileOpDispatch`` and the dispatcher entry point.
- ``thread_scope.h`` defines backend-qualified ``tirx::ThreadScope`` enum tags
  used by layouts, tile Calls, and dispatch contexts. Backends attach explicit
  hierarchy ranks through enum attributes.
- ``context.h`` defines the backend-neutral ``tile::DispatchContext`` and its
  allocation and initialization callbacks.

Their implementations live in ``src/tirx/tile/``. CUDA-specific scope bindings,
active-thread execution contexts, and predicate analysis remain private to
``src/backend/cuda/tile/``. Generic tensor layout types remain in
``include/tvm/tirx/layout.h``.

``tirx.TileDispatch`` runs before ``LowerTensorLayouts`` in the generic compilation pipeline.
Backends may instead follow dispatch with their own layout-lowering pass.
The pass recognizes ``Evaluate(Call)`` by the presence of the operator's
``FTileOpDispatch`` attribute. It validates the call, reads its reflected
``ThreadScope`` ``scope`` attribute, checks its backend against the target,
resolves the scope against the active thread set,
constructs a ``DispatchContext``, and
invokes the instruction's registered lowerer. It replaces the Evaluate with
the returned statement and lowers any nested tensor calls. A lowerer may return
a single statement or a ``SeqStmt`` containing multiple statements.

This preserves launch parameters, inter/intra scope maps, value ranges,
descriptor caches, host initialization, and allocation/initialization
callbacks. Lowering occurs at a statement boundary because one instruction
may require address preparation, layout expansion, and multiple occurrences
of the same core instruction. A residual-call verifier rejects any remaining
tile instruction after expansion using the same dispatch attribute.

The dispatch table is the native Op attribute registry. Python registers
callbacks on individual Ops; C++ implementations can register the same typed
attribute directly. The pass does not call a global Python dispatcher or
consult a Python instruction table to identify tile Ops or resolve their scope.

There is one lowerer per instruction identity. Unsupported layouts and scopes
produce an error that includes the instruction, target, scope, and original
cause. There is no cross-instruction priority search or forced variant bag.

The transient Python ``TensorCall`` view supplies semantic operand names to
existing backend emission code. It is not a registered IR node and is never
serialized. Its decoded options are private lowering data. Every expression
in the original IR resides in ``Call.args``; shared traversal does not inspect
attributes for hidden expressions.

The TVMScript printer validates tensor Calls and prints canonical backend
names, tensor regions, nested tuple operands, and static qualifiers. Parsing
reconstructs the fixed signature. Private Trainium workspace allocation
rewrites Call argument slots before instruction lowering.

See :doc:`../tile_ops` for instruction contracts and caller-owned
algorithms, and :doc:`../api/tile_dispatch` for registration.
