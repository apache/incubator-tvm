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


TVM Errors
==========

TVM may raise errors from Python code, from C++ code reached through the
FFI, or from generated runtime modules.  Error messages usually include
a Python stack trace, and may also include a C++ stack trace when the
error crosses the TVM FFI boundary.

Some errors report invalid user input, unsupported operators, missing
runtime features, or unavailable hardware.  Others report a failed
internal check, usually raised by ``TVM_FFI_ICHECK`` or
``TVM_FFI_THROW`` in C++ code.  Internal check failures often indicate
that TVM reached a state that the implementation did not expect.

What to Check First
-------------------

- Make sure the TVM Python package and native libraries come from the
  same build.  A common symptom of a mismatched environment is importing
  Python files from one checkout while loading ``libtvm`` from another.
- Check that the required runtime is enabled in the CMake configuration.  For
  example, CUDA tests and CUDA compilation require a TVM build with
  CUDA support enabled.
- Check that the target hardware is available to the process.  GPU
  tests may be skipped or fail if the device is not visible inside the
  current container or environment.
- If the error occurs while importing or converting a model, reduce the
  input to the smallest model, operator, or shape that reproduces the
  issue.

Reporting an Issue
------------------

Search the `Apache TVM Discuss Forum <https://discuss.tvm.apache.org/>`_
and the `TVM issue tracker <https://github.com/apache/tvm/issues>`_
for the exact error message first.  If you do not find an existing
report, include the following details when starting a new discussion or
filing an issue:

- The TVM version or git commit hash.
- The Python version, operating system, and hardware.
- The target and runtime being used, such as LLVM, CUDA, Vulkan, or RPC.
- The relevant build configuration from ``build/CMakeCache.txt``.
- A minimal script, model, input shape, or IR module that reproduces the
  failure.
- The full error message, including both Python and C++ stack traces
  when present.

Developer Notes
---------------

For guidance on raising typed errors from TVM code, see
:ref:`error-handling-guide`.  That guide covers when to use specific
error types, how C++ error prefixes map to Python exceptions, and how
``TVM_FFI_ICHECK`` interacts with TVM's error handling.
