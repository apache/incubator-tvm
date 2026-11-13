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

Direct PTX Instructions
=======================

The CUDA backend makes a table-driven PTX instruction namespace available
when authoring TIRx with TVMScript.  This is an authoring namespace, not a
separate IR dialect::

   from tvm_tirx.script import tirx as Tx
   Tx.ptx.add.rn.f32(dst, lhs, rhs)

Each ``Tx.ptx`` call must match a form registered in
:mod:`tvm_tirx.backend.cuda.ptx.table`.  The table validates the instruction family,
modifier combination, operand count, and operand types while constructing one
``tirx.ptx.*`` IR call.  A new instruction is added by extending the table and,
when it needs a new representation, its code-generation support.

This layer is below the layout-aware ``Tx.cuda.tile.*`` / ``Tx.trn.tile.*`` primitives and does not run
tile-op dispatch.

Supported instruction families
------------------------------
The current instruction table has 506 entries across the following 101
families.  These are the Python attribute names written after ``Tx.ptx``; each
family accepts only the modifier and operand combinations declared by its table
entries.

.. hlist::
   :columns: 4

   * ``abs``
   * ``activemask``
   * ``add``
   * ``and_``
   * ``applypriority``
   * ``atom``
   * ``bar``
   * ``barrier``
   * ``bfe``
   * ``bfi``
   * ``bfind``
   * ``bmsk``
   * ``brev``
   * ``clmad``
   * ``clusterlaunchcontrol``
   * ``clz``
   * ``cnot``
   * ``copysign``
   * ``cos``
   * ``cp``
   * ``createpolicy``
   * ``cvt``
   * ``cvt_pack``
   * ``cvta``
   * ``discard``
   * ``div``
   * ``dp2a``
   * ``dp4a``
   * ``elect_sync``
   * ``ex2``
   * ``fabric``
   * ``fence``
   * ``fma``
   * ``fns``
   * ``getctarank``
   * ``griddepcontrol``
   * ``isspacep``
   * ``ld``
   * ``ldmatrix``
   * ``ldu``
   * ``lg2``
   * ``lop3``
   * ``mad``
   * ``mad24``
   * ``mapa``
   * ``match``
   * ``max``
   * ``mbarrier``
   * ``min``
   * ``mma``
   * ``mov``
   * ``movmatrix``
   * ``mul``
   * ``mul24``
   * ``multimem_cp``
   * ``multimem_ld_reduce``
   * ``multimem_red``
   * ``multimem_red_async``
   * ``multimem_st``
   * ``multimem_st_async``
   * ``neg``
   * ``not_``
   * ``or_``
   * ``popc``
   * ``prefetch``
   * ``prefetchu``
   * ``prmt``
   * ``rcp``
   * ``red``
   * ``red_async``
   * ``redux_sync``
   * ``rem``
   * ``rsqrt``
   * ``sad``
   * ``selp``
   * ``set``
   * ``setmaxnreg``
   * ``setp``
   * ``shf``
   * ``shfl_sync``
   * ``shl``
   * ``shr``
   * ``sin``
   * ``slct``
   * ``spcompress``
   * ``spdecompress``
   * ``sqrt``
   * ``st``
   * ``st_async``
   * ``st_bulk``
   * ``stmatrix``
   * ``sub``
   * ``szext``
   * ``tanh``
   * ``tcgen05``
   * ``tensormap_cp_fenceproxy``
   * ``tensormap_replace``
   * ``testp``
   * ``vote_sync``
   * ``wgmma``
   * ``xor``

The generated ``tvm/script/tirx.pyi`` stub provides editor completion for the
valid modifier names and call signatures.  The instruction table is the
complete source of modifier domains, operand layouts, and combination
constraints.

Instruction forms
-----------------
Attribute chains fill PTX modifier slots.  Python keywords have a trailing
underscore, and PTX's ``::`` separator is written as a double underscore::

   Tx.ptx.ld.acquire.gpu.global_.b32(value, pointer)
   Tx.ptx.mbarrier.arrive.shared__cluster.b64(barrier, count)

The indexed form accepts the exact PTX spelling of a registered form and uses
the same table lookup::

   Tx.ptx["st.weak.shared::cta.b32"](pointer, value)

Destination registers are leading operands, so PTX calls are statements.
Predication is keyword-only::

   Tx.ptx.ld.global_.b32(value, pointer, pred=predicate, preserve_dst=True)

``Tx.ptx.pred(value)`` marks a register operand whose PTX type is ``.pred``;
``Tx.ptx.addr(base, byte_offset)`` forms an immediate-offset address for
instructions that accept one; and ``Tx.ptx.SINK`` represents PTX's sink operand
``_`` in table-marked positions.  The generated ``tvm/script/tirx.pyi`` stub
provides editor completion for ``addr``, registered instruction families, and
modifier chains.  ``pred`` and ``SINK`` are available at runtime but are not
declared in the current stub.

PTX versus CUDA helpers
-----------------------
Use ``Tx.ptx`` for one table-described PTX instruction.  Use ``Tx.cuda`` for
backend helpers that require multiple statements, C/C++ expressions, descriptor
packing, or other behavior that cannot be represented as one PTX instruction.
``Tx.ptx_legacy`` preserves operation construction for downstream emitters.
TVM CUDA codegen does not emit these operations; use ``Tx.ptx`` in new kernels.

Namespace API
-------------
.. autoclass:: tvm_tirx.backend.cuda.ptx.PTXNamespace
   :members:
   :exclude-members: SINK
   :special-members: __getitem__

.. automodule:: tvm_tirx.backend.cuda.ptx
   :members:
   :exclude-members: PTXNamespace
