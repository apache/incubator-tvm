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

Tensor Instruction Authoring API
================================

Use ``T.cuda.tile`` or ``T.trn.tile`` after ``from tvm.script import tirx as T``.
The signatures below list runtime operands first and static keyword qualifiers
after ``*``. Tensor variables are expanded to full regions. Trainium unary sources
may default to the destination; its named workspace arguments are optional.

See :doc:`../tile_ops` for memory, scope, descriptor, and synchronization
contracts. These calls produce ordinary IR Calls; the raw scalar instruction
namespaces remain ``T.ptx`` and ``T.nki``.

CUDA
----

.. py:function:: T.cuda.tile.apply(instruction, *operands, scope='thread', **parameters)

   Apply an exact ``T.ptx[opcode]`` to canonical ``group × repeat × atom``
   tensor views. Keyword parameters are serialized Call arguments. The :doc:`../tile_ops` instruction pages list every parameter, default,
   canonical operand image and code-generation rule. Common scope/region
   conventions are in :doc:`../cuda_tile_patterns`. Shared matrix descriptors
   for tcgen05.cp/mma are obtained from the operand layouts.

.. py:function:: T.cuda.tile.ewise(function, dst, *inputs, scope='thread')

   Inline a typed scalar TIRx function at every register coordinate. Tensor
   inputs have the destination shape; scalar inputs broadcast. There is no
   implicit memory staging or packed arithmetic selection.

TRN
---

.. py:function:: T.trn.tile.load(dst, src, *, scope='thread')

.. py:function:: T.trn.tile.store(dst, src, *, scope='thread')

.. py:function:: T.trn.tile.tensor_copy(dst, src, *, scope='thread', max_inst_size=512)

.. py:function:: T.trn.tile.matmul(D, A, B, C=None, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0, acc_psum=None, *, scope='thread')

.. py:function:: T.trn.tile.reciprocal(dst, src, *, scope='thread', max_inst_size=512)

.. py:function:: T.trn.tile.memset(dst, src, *, scope='thread', max_inst_size=512)

.. py:function:: T.trn.tile.activation(dst, src, scale=1.0, bias=0.0, const_bias=None, *, scope='thread', opcode='exp', max_inst_size=512)

.. py:function:: T.trn.tile.tensortensor(dst, lhs, rhs, *, scope='thread', opcode='add', max_inst_size=512)

.. py:function:: T.trn.tile.tensorscalar(dst, lhs, rhs, *, scope='thread', opcode='add', max_inst_size=512)

.. py:function:: T.trn.tile.tensorreduce(dst, src, partial_reduce=None, *, scope='thread', reduce_op='sum', axes=[-1], negate=False, max_inst_size=None)

.. py:function:: T.trn.tile.scalar_tensor_scalar(dst, data, operand0, operand1, *, scope='thread', op0='mul', op1='add', reverse1=False, max_inst_size=512)

.. py:function:: T.trn.tile.scalar_tensor_tensor(dst, data, operand0, operand1, *, scope='thread', op0='mul', op1='add', reverse1=False, max_inst_size=512)

.. py:function:: T.trn.tile.tensorscalar_reduce(dst, reduced, lhs, rhs, partial_reduce=None, *, scope='thread', opcode='add', reduce_op='sum', axes=[-1], max_inst_size=None)

.. py:function:: T.trn.tile.activation_reduce(dst, reduced, src, scale=1.0, bias=0.0, const_bias=None, partial_reduce=None, *, scope='thread', opcode='exp', reduce_op='sum', axes=[-1], max_inst_size=None)

.. py:function:: T.trn.tile.affine_select(dst, true_value, false_value, pred, *, scope='thread')
