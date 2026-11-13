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

tvm_tirx.runtime
----------------
.. automodule:: tvm_tirx.runtime
   :members:

tvm.runtime
-----------

Benchmarking belongs to the parent runtime. Use
``tvm.runtime.time_evaluator(module, func_name, device, ...)`` with an ordinary
``tvm_ffi.Module`` and either a local device or an RPC remote device. The returned
callable reports a ``BenchmarkResult`` containing the repeated measurements.

.. autofunction:: tvm.runtime.time_evaluator

.. autoclass:: tvm.runtime.BenchmarkResult
   :members:
