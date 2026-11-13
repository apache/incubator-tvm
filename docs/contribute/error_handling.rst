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

.. _error-handling-guide:

Error Handling Guide
====================

.. contents::
  :depth: 2
  :local:

TVM uses Python exception types to indicate specific types of error.
Please raise a specific error type when possible, so that users can
write code to handle a specific error category if necessary.
You can directly raise the specific error object in python.
In other languages like c++, you simply add ``<ErrorType>:`` prefix to
the error message(see below).

.. note::

   Prefer built-in Python exceptions for common error categories.

Raise a Specific Error in C++
-----------------------------
You can add ``<ErrorType>:`` prefix to your error message to
raise an error of the corresponding type.
Note that you do not have to add a new type
``RuntimeError`` will be raised by default when
there is no error type prefix in the message.
This mechanism works for both ``LOG(FATAL)`` and ``TVM_FFI_ICHECK`` macros.
The following code gives an example on how to do so.

.. code:: c

  // src/api_test.cc
  void ErrorTest(int x, int y) {
    TVM_FFI_ICHECK_EQ(x, y) << "ValueError: expect x and y to be equal."
    if (x == 1) {
      LOG(FATAL) << "InternalError: cannot reach here";
    }
  }

When a C++ function registered via the FFI raises an error with a typed prefix,
the TVM FFI system will automatically map it to the corresponding Python exception
class. For example, a ``ValueError:`` prefix in the error message will raise a Python
``ValueError``, and an ``InternalError:`` prefix will raise ``RuntimeError``.

TVM's FFI system combines both the Python and C++ stacktraces into a single message,
and uses ``RuntimeError`` for error kinds without a registered Python exception.


How to choose an Error Type
---------------------------
You can go through the error types are listed below, try to use common
sense and also refer to the choices in the existing code.
We try to keep a reasonable amount of error types.
If you feel there is a need to add a new error type, do the following steps:

- Send a RFC proposal with a description and usage examples in the current codebase.
- Register the Python exception with ``tvm_ffi.register_error`` and document its use.
- Update the list in this file to include the new error type.
- Change the code to use the new error type.

We also recommend to use less abstraction when creating the short error messages.
The code is more readable in this way, and also opens path to craft specific
error messages when necessary.

.. code:: python

   def preferred():
       # Very clear about what is being raised and what is the error message.
       raise NotImplementedError("Operator relu is not implemented in the ONNX frontend")

   def _op_not_implemented(op_name):
       return NotImplementedError("Operator {} is not implemented.").format(op_name)

   def not_preferred():
       # Introduces another level of indirection.
       raise _op_not_implemented("relu")

If we need to introduce a wrapper function that constructs multi-line error messages,
please put wrapper in the same file so other developers can look up the implementation easily.
