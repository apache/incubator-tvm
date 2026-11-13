.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements. See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership. The ASF licenses this file
   to you under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

CUDA backend configuration
==========================

Set the nested mapping on ``Target.backend_config`` and pass that Target to
``tvm.compile``. Individual kernels can override it with
``T.device_entry(..., backend_config=...)``. Each backend owns its configuration;
CUDA currently provides these fixed keys:

.. list-table:: CUDA keys
   :header-rows: 1
   :widths: 20 30 50

   * - Key
     - Type
     - Meaning
   * - ``arch``
     - ``str``
     - One real architecture, such as ``sm_100a``; used during lowering and compilation.
   * - ``compiler``
     - ``"nvcc"`` or ``"nvrtc"``
     - CUDA frontend; defaults to NVRTC.
   * - ``target_format``
     - ``"ptx"``, ``"cubin"``, or ``"fatbin"``
     - NVRTC supports PTX and cubin; NVCC also supports fatbin.
   * - ``nvcc``
     - ``list[str]``
     - Native NVCC arguments; defaults to ``["--use_fast_math"]``.
   * - ``nvrtc``
     - ``list[str]``
     - Native NVRTC arguments; defaults to ``["--use_fast_math"]``.
   * - ``ptxas``
     - ``list[str]``
     - Native assembler arguments forwarded through the selected frontend.
       Defaults to ``["-v", "--warn-on-local-memory-usage", "--register-usage-level=10"]``.

``BackendConfig`` is a ``TypedDict`` for completion and static typing. Its
constructor returns an ordinary dictionary; all keys are optional. Compilation
and entry construction validate and snapshot the supplied mapping. Native
compiler arguments are not modeled as separate Python fields.

.. code-block:: python

    from tvm_tirx.backend.cuda import BackendConfig

    cuda_config = BackendConfig(
        compiler="nvrtc",
        nvrtc=["--use_fast_math", "--ftz=false"],
    )
    target = tvm_tirx.target.Target({
        "kind": "cuda", "arch": "sm_100a",
        "backend_config": {"cuda": cuda_config},
    })
    executable = tvm.compile(func, target=target)

.. autoclass:: tvm_tirx.backend.cuda.BackendConfig
   :members:
   :undoc-members:

.. autoclass:: tvm_tirx.backend.cuda.compiler.CompilationResult
   :members:

.. autofunction:: tvm_tirx.backend.cuda.compiler.compile_source

See :doc:`../native_basics/cuda/compiling` for target-tag defaults, inheritance,
per-entry compilation, and artifact behavior. See :doc:`cuda_compile_options`
for native NVCC/NVRTC/ptxas argument usage and official compiler references.
