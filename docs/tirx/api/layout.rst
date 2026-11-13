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

Layout IR
=========

Layouts describe how logical tensor coordinates map to storage and execution
axes.  See :doc:`../layout` for the programming model and worked examples.

The primary classes are re-exported from ``tvm_tirx.tirx``, but are defined in this
module:

.. autoclass:: tvm_tirx.tirx.layout.Layout
   :members:
   :no-index:

.. autoclass:: tvm_tirx.tirx.layout.Axis
   :members:
   :no-index:

.. autoclass:: tvm_tirx.tirx.layout.Iter
   :members:
   :no-index:

.. autoclass:: tvm_tirx.tirx.layout.TileLayout
   :members:
   :no-index:

.. autoclass:: tvm_tirx.tirx.layout.ComposeLayout
   :members:
   :no-index:

``S[...]`` and ``R[...]`` build shard and replica layout specifications,
respectively.  Named axes such as ``laneid``, ``warpid``, ``tid_in_wg``,
``TLane``, and ``TCol`` belong to CUDA. Use ``T.cuda.Axis.laneid`` or
import axes from ``tvm_tirx.backend.cuda.axis``. Trainium's ``P``, ``F``, and
``Bank`` belong to ``T.trn.Axis``. The generic memory axis remains ``T.Axis.m``.

CUDA memory/instruction layout factories are defined in
``tvm_tirx.backend.cuda.layout`` and exposed as ``T.cuda.layout``. Trainium
layouts use explicit partition, free and bank axes. For example::

   from tvm_tirx.backend.trn.axis import P, F, Bank
   from tvm_tirx.tirx.layout import TileLayout, S

   # A (128, 512) SBUF tile: partition p, free element f.
   sbuf_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])
   # Two (128, 512) PSUM tiles, one per bank.
   psum_layout = TileLayout(S[(2, 128, 512) : (1 @ Bank, 1 @ P, 1 @ F)])

Annotation strings are not accepted by generic tensor constructors.

Omitting ``layout`` or passing ``layout=None`` leaves every storage scope
without a layout. ``layout="default"`` explicitly requests a compact
``TileLayout``. Ordinary views preserve shape, strides and element offsets
without creating a layout. ``TensorType.effective_layout()`` returns an
explicit layout or infers a temporary compact layout for a global tensor.
Tile instructions query it through ``tensor.ty.effective_layout()``, using
the full tensor shape before slicing an operand region. The query does not
change the tensor type or its ``layout=None`` field. Noncompact global tensors
and other storage scopes require an explicit layout when the instruction uses one.

.. automodule:: tvm_tirx.tirx.layout
   :members:
   :no-index:
   :exclude-members: Axis, ComposeLayout, Iter, Layout, TileLayout
