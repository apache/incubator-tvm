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

Trainium Authoring and Support APIs
===================================

NKI authoring namespace
-----------------------

The Trainium backend installs ``Tx.nki``.  Its current operations are
``load``, ``store``, ``tensor_copy``, ``matmul``, ``activation``,
``activation_reduce``, ``reciprocal``, ``tensorreduce``, ``tensortensor``,
``tensorscalar``, ``tensorscalar_reduce``, ``scalar_tensor_tensor``,
``scalar_tensor_scalar``, ``memset``, ``identity``, and ``affine_select``.

.. autoclass:: tvm_tirx.backend.trn.script.NKINamespace
   :members:
   :no-index:

Layout helpers
--------------

.. automodule:: tvm_tirx.backend.trn.layout
   :members:
   :no-index:

Compilation pipeline
--------------------

.. automodule:: tvm_tirx.backend.trn.pipeline
   :members:
   :no-index:

Transforms
----------

.. automodule:: tvm_tirx.backend.trn.transform
   :members:
   :no-index:

.. autoclass:: tvm_tirx.backend.trn.transform.TrnNaiveAllocator
   :members:
   :no-index:

.. autoclass:: tvm_tirx.backend.trn.transform.TrnPrivateTensorAlloc
   :members:
   :no-index:
