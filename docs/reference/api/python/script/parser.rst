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

tvm_tirx.script.parser
----------------------

tvm_tirx.script.parser
**********************
.. automodule:: tvm_tirx.script.parser
   :members:
   :imported-members:

tvm_tirx.script.parser.jit_support
**********************************
These helpers carry ``const_args``, the selected fixed argument map, into generated builder
execution. Parameter validation and caching remain with the JIT entry point.

.. automodule:: tvm_tirx.script.parser.jit_support
   :members: use_specialization, read_specialization_bindings

tvm_tirx.script.parser.protocol_registry
****************************************
Language variants register source syntax policies independently of their
construction hooks. Namespace objects and lazy initialization callbacks are
registered through :func:`tvm_tirx.script.parser.register_namespace` and
:func:`tvm_tirx.script.parser.register_namespace_initializer`.

.. automodule:: tvm_tirx.script.parser.protocol_registry
   :members: register_scalar_annotation, register_mutable_decl

The language variant aliases below share the public construction namespaces documented
in :doc:`script`. Parser entry points above use the canonical frontend.

tvm_tirx.script.parser.ir
*************************
.. automodule:: tvm_tirx.script.parser.ir

tvm_tirx.script.parser.relax
****************************
.. automodule:: tvm_tirx.script.parser.relax

tvm_tirx.script.parser.tirx
***************************
.. automodule:: tvm_tirx.script.parser.tirx
