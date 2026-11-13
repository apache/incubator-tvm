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

Core TIRx IR
============

This page lists the core IR nodes and scalar operations in the top-level
namespace.  Layouts, execution scopes, visitors, compilation helpers, and
tile-dispatch extensions are documented on their focused pages and excluded
here so the same objects are not expanded twice.

For C++ construction, include ``tvm/tirx/expr.h`` for ``TensorVar``, tensor
loads, and tensor-region constructors. Include ``tvm/tirx/type.h`` for
``TensorType`` and ``TensorMapType``. Tensor regions use the shared
``TensorRegion`` expression from ``tvm/ir/expr.h`` and ``TensorRegionType``
from ``tvm/ir/type.h``.

.. automodule:: tvm_tirx.tirx
   :members:
   :imported-members:
   :exclude-members: Call, CommReducer, ComposeLayout, DispatchContext,
      ThreadScope, Expr, Layout, Op, PrimExpr, Reduce,
      TileLayout, Var, build, const,
      get_default_tir_pipeline, get_tir_pipeline, register_tir_pipeline
