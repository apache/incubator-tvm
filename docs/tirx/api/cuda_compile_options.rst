.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements. See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership. The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

Native CUDA compiler arguments
==============================

CUDA uses six fixed configuration keys; see :doc:`cuda_compile`. Pass other
compiler settings as native arguments in the ``nvcc``, ``nvrtc``, or ``ptxas``
lists:

.. code-block:: python

    backend_config = {"cuda": {
        "arch": "sm_100a",
        "compiler": "nvcc",
        "nvcc": ["--use_fast_math", "--ftz=false", "--std=c++20", "-I/include"],
        "nvrtc": ["--use_fast_math", "--ftz=false", "--std=c++20", "-I/include"],
        "ptxas": ["-O3", "--register-usage-level=10"],
    }}

Each list contains argv items, without shell parsing. The selected frontend
receives its own list and the forwarded ``ptxas`` list. Each list replaces the
inherited list completely: ``nvrtc=[]`` clears the default fast-math flag and
``ptxas=[]`` clears the default assembler arguments.

TVM validates fixed keys and types. Architecture, output format, and toolchain
routing are controlled by the corresponding configuration keys and the compiler
adapter. Native flags that override those controls are reserved. The selected
compiler validates other arguments, including their values and version support;
new compiler flags can be used directly through these lists.

TVM adds the integration arguments needed for CUDA headers, architecture, output
retrieval, and NVSHMEM device linking. Supported outputs are PTX, cubin, and NVCC
fatbin. Additional artifact or linking workflows require backend support beyond
passing their compiler flags.

Consult the `NVCC compiler options`_ and `NVRTC compilation options`_ for the
installed toolkit version. Use ``ptxas --help`` to inspect its assembler options.

.. _NVCC compiler options: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
.. _NVRTC compilation options: https://docs.nvidia.com/cuda/nvrtc/index.html
