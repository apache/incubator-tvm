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

Backend Extension API
=====================

Backend loading
---------------

``tvm_tirx.backend`` discovers and loads target-owned Python semantics.  Depending
on the target, loading a backend registers its TVMScript namespaces,
tile-dispatch implementations, target tags, compilation-pipeline entry points,
and code-generation support.

.. automodule:: tvm_tirx.backend
   :members:
   :no-index:

CUDA registration
-----------------

.. autofunction:: tvm_tirx.backend.cuda.register_backend
   :no-index:

.. autofunction:: tvm_tirx.backend.cuda.script_namespace
   :no-index:

.. autofunction:: tvm_tirx.backend.cuda.script_namespaces
   :no-index:

Trainium registration
---------------------

.. autofunction:: tvm_tirx.backend.trn.register_backend
   :no-index:

.. autofunction:: tvm_tirx.backend.trn.script_namespace
   :no-index:

.. autofunction:: tvm_tirx.backend.trn.script_namespaces
   :no-index:

These functions are backend integration points.  Kernel-facing CUDA APIs are
documented in :doc:`cuda` and :doc:`ptx`; Trainium APIs are documented in
:doc:`trainium`.  See :doc:`../arch/backends` for module ownership and
registration side effects.
