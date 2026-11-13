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

.. _external-library-dispatch:

External Library Dispatch (BYOC)
================================

When deploying models, certain operator patterns (e.g., matmul + bias + relu) can be executed
more efficiently by vendor-optimized libraries such as cuBLAS, CUTLASS, cuDNN, or DNNL. TVM's
**BYOC (Bring Your Own Codegen)** mechanism identifies these patterns in a Relax module and
offloads them to external backends, while keeping the rest of the computation on TVM's own
generated kernels.

This document explains the BYOC pipeline: how patterns are registered, how subgraphs are
matched and extracted, how backend code generators are invoked, and how the externally compiled
code is executed at runtime.


Overview
--------

The BYOC pipeline consists of four stages:

.. code-block:: text

   IRModule (high-level Relax IR)
        │
        ▼  FuseOpsByPattern              ← match high-level ops, create composite functions
   IRModule (with Composite + Codegen attributes)
        │
        ▼  RunCodegen                    ← invoke backend codegen via FFI
   IRModule (with call_dps_packed to ExternFunc)
   + external runtime Modules
        │
        ▼  LegalizeOps + FuseOps + ...   ← compile remaining ops normally
        │
        ▼  VM compilation                ← link external modules into executable
   Deployable artifact

Each stage is a Relax transformation pass that operates on the ``IRModule``:

1. **FuseOpsByPattern** — matches operator subgraphs against registered patterns and groups them
   into composite functions annotated with ``Composite`` and ``Codegen`` attributes.
2. **MergeCompositeFunctions** (optional) — merges multiple composite functions targeting the same
   backend when inter-operator dependencies allow.
3. **RunCodegen** — finds all functions with a ``Codegen`` attribute, invokes the corresponding
   backend code generator via FFI, and replaces the original calls with ``call_dps_packed``
   to externally compiled functions.
4. **Linking** — the resulting external ``tvm_ffi.Module``\ s are attached to the ``IRModule``
   as the ``external_mods`` attribute and bundled into the final executable during
   ``relax.build()``.


Pattern Registration
--------------------

Each backend registers the operator patterns it supports in a **global pattern registry**
(``python/tvm/relax/backend/pattern_registry.py``). The registry is a static table that maps
pattern names to ``FusionPattern`` objects.

Registering patterns
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tvm.relax.backend.pattern_registry import register_patterns
   from tvm.relax.backend.patterns import make_matmul_pattern

   register_patterns([
       (
           "cublas.matmul",              # pattern name (prefix = backend)
           *make_matmul_pattern(          # returns (DFPattern, annotation_patterns)
               with_bias=False,
           ),
           _check_matmul,                # check function
       ),
       (
           "cublas.matmul_bias_relu",
           *make_matmul_pattern(
               with_bias=True,
               activation="relax.nn.relu",
           ),
           _check_matmul,
       ),
       # ... more patterns
   ])

Each entry is a tuple of ``(name, pattern, annotation_patterns, check_func)`` that gets
converted to a ``FusionPattern`` object. The name prefix (e.g., ``"cublas"``) identifies the
backend; ``get_patterns_with_prefix("cublas")`` retrieves all patterns for that backend.

Patterns registered later have **higher priority** — when a subgraph matches multiple patterns,
the highest-priority match wins.

Pattern templates
~~~~~~~~~~~~~~~~~

``python/tvm/relax/backend/patterns.py`` provides reusable templates for common patterns:

- ``make_matmul_pattern(with_bias, activation, transposed_rhs)`` — matmul with optional bias
  and activation fusion
- ``make_conv2d_pattern(with_bias, activation)`` — 2D convolution
- ``make_attention_pattern()`` — multi-head attention
- ``make_residual_block_pattern()`` — residual connections
- ``make_layer_norm_pattern()`` / ``make_rms_norm_pattern()`` — normalization layers

Each template returns ``(DFPattern, Mapping[str, DFPattern])`` — the main pattern and its
annotation sub-patterns.

Check functions
~~~~~~~~~~~~~~~

The check function validates whether a matched subgraph can actually be handled by the backend.
It receives a ``PatternCheckContext`` and returns ``True`` to accept or ``False`` to reject.

Typical checks include:

- **Data type support**: verify the operand dtypes are supported (e.g., cuBLAS supports
  float16, float32, int8, bfloat16, float8 for matmul).
- **Shape constraints**: verify reduction axes are constant, batch dimensions are compatible.
- **Leaking intermediates**: reject if an intermediate result is used outside the fused group
  (via ``has_leaking_intermediate_variables()``).


Partitioning
------------

After patterns are registered, a backend provides a **partition function** that applies
``FuseOpsByPattern`` to an ``IRModule``:

.. code-block:: python

   # python/tvm/relax/backend/cuda/cublas.py
   def partition_for_cublas(mod, bind_constants=False):
       patterns = get_patterns_with_prefix("cublas")
       return transform.FuseOpsByPattern(
           patterns, bind_constants=bind_constants, annotate_codegen=True
       )(mod)

With ``annotate_codegen=True``, each matched subgraph is wrapped in a two-level function
structure:

.. code-block:: text

   # Outer function — tagged for the codegen backend
   @R.function
   def fused_relax_matmul_cublas0(args...):
       R.func_attr({"Codegen": "cublas", "global_symbol": "fused_relax_matmul_cublas0"})
       ...
           # Inner function — identifies the specific pattern
           @R.function(private=True)
           def composite(args...):
               R.func_attr({"Composite": "cublas.matmul_bias_relu"})
               lv0 = R.matmul(x, w)
               lv1 = R.add(lv0, bias)
               lv2 = R.nn.relu(lv1)
               return lv2
       ...

The outer function carries the ``Codegen`` attribute that ``RunCodegen`` uses to dispatch to the
right backend. The inner function carries the ``Composite`` attribute that the backend codegen
uses to identify which operation to emit.

MergeCompositeFunctions
~~~~~~~~~~~~~~~~~~~~~~~

When ``annotate_codegen=False``, ``FuseOpsByPattern`` only creates inner functions with
``Composite`` attributes. A separate ``MergeCompositeFunctions`` pass then groups multiple
composite functions targeting the same backend into a single outer function with ``Codegen``
and ``global_symbol`` attributes.

This is useful when multiple sequential operations should be sent to the same backend as a
single unit (e.g., a sequence of cuBLAS matmuls that share intermediate results). The pass
checks that merging does not create cyclic dependencies between groups.


Code Generation
---------------

``RunCodegen`` (``src/relax/transform/run_codegen.cc``) is the pass that triggers backend
code generation:

1. Scan the module for all functions with a ``Codegen`` attribute.
2. Group them by backend target name.
3. For each backend, look up the registered codegen function via FFI key
   ``"relax.ext.<backend>"`` (e.g., ``"relax.ext.cublas"``).
4. Call the codegen function, which returns an array of compiled ``tvm_ffi.Module``\ s.
5. Replace the original function calls with ``call_dps_packed(ExternFunc(...), args)``.
6. Attach the compiled modules to the ``IRModule`` as the ``external_mods`` attribute.

Codegen registration
~~~~~~~~~~~~~~~~~~~~

Each backend registers a codegen function via TVM's FFI mechanism:

.. code-block:: cpp

   // src/relax/backend/contrib/cublas/codegen.cc
   ffi::Array<ffi::Module> CublasCompiler(
       ffi::Array<Function> functions,
       ffi::Map<ffi::String, ffi::Any> options,
       ffi::Map<Constant, ffi::String> constant_names) {
     ffi::Array<ffi::Module> compiled_functions;
     for (const auto& func : functions) {
       CublasJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
       serializer.serialize(func);
       auto graph_json = serializer.GetJSON();
       auto names = serializer.GetConstantNames();
       const auto pf = ffi::Function::GetGlobalRequired("runtime.CublasJSONRuntimeCreate");
       compiled_functions.push_back(
           pf(GetExtSymbol(func), graph_json, names).cast<ffi::Module>());
     }
     return compiled_functions;
   }

   TVM_FFI_STATIC_INIT_BLOCK() {
     namespace refl = tvm::ffi::reflection;
     refl::GlobalDef().def("relax.ext.cublas", CublasCompiler);
   }

The codegen function receives:

- ``functions``: the Relax functions with ``Codegen`` attribute to compile.
- ``options``: backend-specific compilation options.
- ``constant_names``: mapping from constant values to their names (for weight handling).

It returns an array of ``tvm_ffi.Module`` objects — one per function — that contain the
externally compiled code.

Codegen strategies
~~~~~~~~~~~~~~~~~~

TVM provides two base classes for implementing backend codegens:

- **JSONSerializer** (``src/relax/backend/contrib/codegen_json/codegen_json.h``): serializes the
  composite function into a JSON graph representation. At runtime, a backend-specific JSON
  runtime module interprets the graph and dispatches to library calls. Used by cuBLAS, cuDNN,
  and most backends.

- **CSourceCodegen** (``src/relax/backend/contrib/codegen_c/codegen_c.h``): generates C/CUDA
  source code that is compiled and linked. Used when the backend requires ahead-of-time
  compilation.


Runtime Execution
-----------------

After ``RunCodegen``, the original high-level function calls are replaced with:

.. code-block:: python

   R.call_dps_packed(ExternFunc("fused_relax_matmul_cublas0"), (x, w, bias), ...)

At runtime, ``call_dps_packed`` invokes the externally compiled function through the
``PackedFunc`` interface. The external ``tvm_ffi.Module``\ s (produced by the codegen) are
imported into the final executable during ``relax.build()`` and are available via the module's
function lookup mechanism.

For JSON-based backends (cuBLAS, cuDNN), the runtime module deserializes the JSON graph and
dispatches each node to the corresponding library API call. For source-based backends, the
compiled native code is called directly.


Adding a New Backend
--------------------

To add support for a new external library:

1. **Define patterns** in ``python/tvm/relax/backend/<target>/``:

   - Create DFPatterns using templates from ``patterns.py`` or custom patterns.
   - Write check functions to validate dtypes, shapes, and other constraints.
   - Register patterns with ``register_patterns()``.
   - Provide a ``partition_for_<backend>(mod)`` convenience function.

2. **Implement codegen** in ``src/relax/backend/contrib/<target>/``:

   - Subclass ``JSONSerializer`` or ``CSourceCodegen``.
   - Implement the visitor that converts composite functions to the target format.
   - Register the codegen function as ``"relax.ext.<target>"``.

3. **Implement runtime** (for JSON-based backends):

   - Create a JSON runtime module that interprets the serialized graph and dispatches
     to the library's API calls.
   - Register the runtime constructor as ``"runtime.<Target>JSONRuntimeCreate"``.


Supported Backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Backend
     - Patterns
     - Operations
   * - cuBLAS
     - ``cublas.*``
     - Matmul (with bias, activation, transpose, dequantize variants)
   * - CUTLASS
     - ``cutlass.*``
     - Matmul, conv2d, attention, residual blocks, decode matmul
   * - cuDNN
     - ``cudnn.*``
     - Conv2d (NHWC/NCHW), stacked attention
   * - DNNL
     - ``dnnl.*``
     - Matmul, conv2d (x86 CPU). Codegen exists at C++ level; patterns are
       defined in tests rather than pre-registered.


Source Code Map
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Path
     - Contents
   * - ``python/tvm/relax/backend/pattern_registry.py``
     - Pattern registry API (register_patterns, get_patterns_with_prefix)
   * - ``python/tvm/relax/backend/patterns.py``
     - Reusable pattern templates (make_matmul_pattern, etc.)
   * - ``python/tvm/relax/backend/cuda/cublas.py``
     - cuBLAS patterns and partition_for_cublas
   * - ``python/tvm/relax/backend/cuda/cutlass.py``
     - CUTLASS patterns and partition_for_cutlass
   * - ``python/tvm/relax/backend/cuda/cudnn.py``
     - cuDNN patterns and partition_for_cudnn
   * - ``src/relax/backend/pattern_registry.cc``
     - Pattern registry C++ implementation
   * - ``src/relax/transform/run_codegen.cc``
     - RunCodegen pass (CodeGenRunner)
   * - ``src/relax/transform/merge_composite_functions.cc``
     - MergeCompositeFunctions pass
   * - ``src/relax/backend/contrib/cublas/codegen.cc``
     - cuBLAS codegen (JSONSerializer-based)
   * - ``src/relax/backend/contrib/cutlass/codegen.cc``
     - CUTLASS codegen
   * - ``src/relax/backend/contrib/codegen_json/codegen_json.h``
     - JSONSerializer base class
   * - ``src/relax/backend/contrib/codegen_c/codegen_c.h``
     - CSourceCodegen base class
