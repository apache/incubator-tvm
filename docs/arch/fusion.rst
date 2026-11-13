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

.. _fusion-arch:

Operator Fusion
===============

Operator fusion is one of the most impactful optimizations in TVM. Instead of launching one kernel
per operator (e.g., conv2d, bias_add, relu), fusion merges multiple operators into a single kernel,
eliminating intermediate memory allocations and kernel launch overhead.

TVM provides two complementary fusion mechanisms:

- **Automatic fusion** (``FuseOps`` + ``FuseTIR``): groups operators based on their computational
  patterns using a post-dominator analysis algorithm.
- **Pattern-based fusion** (``FuseOpsByPattern``): groups operators that match user-defined
  dataflow patterns, typically for offloading to external backends (cuBLAS, CUTLASS, DNNL, etc.).

Both produce the same output: Relax functions marked with ``Primitive=True`` that are later
lowered to fused TIR kernels or dispatched to external libraries.

Overview
--------

Fusion involves three passes:

.. code-block:: text

   IRModule (after LegalizeOps)
        │
        ▼  AnnotateTIROpPattern        ← label each op (elementwise, reduce, etc.)
   IRModule (annotated)
        │
        ▼  FuseOps                     ← group ops into fused Relax functions
   IRModule (with fused functions marked Primitive=True)
        │
        ▼  FuseTIR                     ← merge TIR Functions inside each group
   IRModule (fused TIR kernels)

In the compilation pipeline, these passes appear in the backend-specific ``legalize_passes``
phase. For example, the CUDA pipeline (``python/tvm/relax/backend/cuda/pipeline.py``) runs:

.. code-block:: python

   LegalizeOps()          # lower Relax ops to call_tir
   AnnotateTIROpPattern() # annotate pattern kinds
   FoldConstant()
   FuseOps()              # group ops
   FuseTIR()              # merge TIR functions


Operator Pattern Classification
-------------------------------

Before fusion, ``AnnotateTIROpPattern`` analyzes each TIR function in the module and assigns
an ``OpPatternKind``. The fusion algorithm uses these pattern kinds to decide which operators
can be fused together.

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Pattern Kind
     - Value
     - Description
   * - ``kElemWise``
     - 0
     - Elementwise: one-to-one input/output mapping (e.g., ``add``, ``relu``, ``exp``).
   * - ``kBroadcast``
     - 1
     - Broadcasting: output axes map to input axes in order, but some input axes may be
       broadcast (e.g., ``bias_add``). Note: ``transpose`` is **not** broadcast because axes
       are reordered.
   * - ``kInjective``
     - 2
     - Injective: each output element depends on a single input element, but the mapping may
       be non-trivial (e.g., ``reshape``, ``concatenate``, ``transpose``).
   * - ``kCommReduce``
     - 3
     - Communicative reduction: output elements aggregate over input elements
       (e.g., ``sum``, ``max``, ``mean``).
   * - ``kOutEWiseFusable``
     - 4
     - Complex operation whose output can accept elementwise followers, but cannot chain
       with another complex op (e.g., ``conv2d``, ``matmul``, ``dense``).
   * - ``kTuple``
     - 7
     - Tuple node. Can fuse into subsequent injective ops but is treated specially.
   * - ``kOpaque``
     - 8
     - Opaque: cannot be fused (e.g., external function calls, operations with side effects).

These kinds form an ordering: lower values are "simpler" and more fusable. The fusion algorithm
uses ``CombinePattern(lhs, rhs) = max(lhs, rhs)`` when merging patterns along a path.


FuseOps: Automatic Fusion
-------------------------

``FuseOps`` (``src/relax/transform/fuse_ops.cc``) groups bindings in a dataflow block into
new Relax functions. It operates only within ``DataflowBlock``\ s — if your module doesn't have
any, run ``ConvertToDataflow`` first.

Algorithm
~~~~~~~~~

The fusion algorithm addresses diamond-shaped dataflow branches, where a single producer
(e.g., conv2d) has multiple consumers that eventually reconverge:

.. code-block:: text

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add

At the point of ``conv2d``, we don't know if all future paths will merge. The algorithm uses
**post-dominator analysis** to resolve this:

1. **Build forward graph**: construct an ``IndexedForwardGraph`` from the dataflow block.
   Each node has an ``OpPatternKind`` and a list of forward edges.

2. **Build post-dominator tree**: compute the immediate post-dominator of each node using
   Least Common Ancestor (LCA) on the DAG. The post-dominator of a node is the closest
   downstream node where **all** future paths converge.

3. **Fuse groups**: for each node in topological order, check if it can be fused with its
   immediate post-dominator:

   - **CheckPath**: verify that all paths from the node to its post-dominator satisfy the
     fusion conditions (pattern compatibility, depth limits, argument limits).
   - **CommitFuse**: mark all intermediate nodes as belonging to the same group using a
     Union-Find data structure.

4. **Create grouped functions**: extract each group into a new ``relax.Function`` with the
   attribute ``Primitive=True``. Replace the original bindings with a call to the grouped
   function.

Fusion rules
~~~~~~~~~~~~

The key fusion decisions depend on the ``OpPatternKind`` of the source, the path, and the
post-dominator. The algorithm runs in three phases (via ``GraphPartitioner::RunFuse``) so that
higher-complexity ops get a chance to fuse first:

- **Phase 0**: ``kOutEWiseFusable`` ops (e.g., ``conv2d``) can fuse with their elementwise
  post-dominator if all intermediate ops are broadcast or simpler. This enables patterns like
  conv2d + bias_add + relu. Two ``kOutEWiseFusable`` ops cannot fuse together.
- **Phase 1**: ``kInjective`` and ``kTuple`` ops can fuse only when all paths to the
  post-dominator are injective or simpler. This is deferred to phase 1 so that
  ``kOutEWiseFusable`` groups are finalized first.
- **Phase 2**: fuse injective ops into intermediate tuple nodes that have already been absorbed
  by subsequent injective groups.

``kElemWise`` / ``kBroadcast`` ops are processed in **every** phase (not restricted to one):
they can fuse into a post-dominator that is injective or reduction. The sink (final node) may
also be a ``kOutEWiseFusable`` group that was formed in phase 0 — this is how elementwise
producers merge into an existing conv2d fusion group.

Additional constraints:

- **Reduction** (``kCommReduce``) ops never initiate fusion — they act as sinks only. Elementwise
  and broadcast producers can fuse *into* a reduction, but a reduction cannot fuse forward.
- **Opaque** ops are fusion barriers.
- A group cannot exceed ``kMaxFusedOps`` (256) nodes or the maximum function argument count.

Example
~~~~~~~

Given two elementwise ops (``add``, ``exp``) and one injective op (``squeeze``).
The examples below are simplified pseudocode — real TVMScript would reference TIR functions
via ``cls.func_name``:

.. code-block:: python

   # Before FuseOps (simplified)
   @R.function
   def main(x: R.Tensor((10, 20), "float32")):
       with R.dataflow():
           lv0 = R.call_tir(add, (x, const_1), ty_args=[R.Tensor((10, 20), "float32")])
           lv1 = R.call_tir(exp, (lv0,), ty_args=[R.Tensor((10, 20), "float32")])
           gv = R.call_tir(squeeze, (lv1,), ty_args=[R.Tensor((10, 20), "float32")])
           R.output(gv)
       return gv

After ``FuseOps``, all three are grouped into a single function:

.. code-block:: python

   # After FuseOps
   @R.function(private=True)
   def fused_add_exp_squeeze(x, p0):
       R.func_attr({"Primitive": True})
       with R.dataflow():
           lv0 = R.call_tir(add, (x, p0), ...)
           lv1 = R.call_tir(exp, (lv0,), ...)
           gv = R.call_tir(squeeze, (lv1,), ...)
           R.output(gv)
       return gv

   @R.function
   def main(x: R.Tensor((10, 20), "float32")):
       with R.dataflow():
           gv = fused_add_exp_squeeze(x, const_1)
           R.output(gv)
       return gv


FuseTIR: Merging TIR Functions
------------------------------

``FuseTIR`` (``src/relax/transform/fuse_tir.cc``) takes the grouped Relax functions produced by
``FuseOps`` and merges their internal TIR ``Function``\ s into a single TIR function.

Before ``FuseTIR``, a fused group still contains multiple ``R.call_tir`` calls to separate
TIR functions. ``FuseTIR`` inlines and merges them:

.. code-block:: text

   Before FuseTIR:
     fused_add_exp_squeeze:
       call_tir(add, ...)        → separate TIR Function
       call_tir(exp, ...)        → separate TIR Function
       call_tir(squeeze, ...)    → separate TIR Function

   After FuseTIR:
     fused_add_exp_squeeze:      → single merged TIR Function

The merged function eliminates intermediate buffers — the output of ``add`` is directly consumed
by ``exp`` without writing to and reading from global memory. This is the core performance benefit
of fusion.

Internally, ``FuseTIR`` uses a ``SymbolicMatcher`` to align symbolic shape variables across the
TIR functions being merged, ensuring that dimensions are correctly mapped when combining buffer
accesses.


FuseOpsByPattern: Pattern-Based Fusion
--------------------------------------

While ``FuseOps`` makes fusion decisions automatically based on operator patterns,
``FuseOpsByPattern`` lets you specify exactly which operator combinations to fuse using
the Relax :ref:`Dataflow Pattern Language (DPL) <relax-dpl>`.

This is primarily used for **backend-specific dispatch**: identifying operator subgraphs that
should be offloaded to external libraries like cuBLAS, CUTLASS, cuDNN, or DNNL.

FusionPattern
~~~~~~~~~~~~~

A ``FusionPattern`` (``python/tvm/relax/transform/transform.py``) defines what to match:

.. code-block:: python

   from tvm.relax.dpl import wildcard, is_op
   from tvm.relax.transform import FusionPattern

   # Match: matmul(x, w) + bias
   x = wildcard()
   w = wildcard()
   bias = wildcard()
   matmul = is_op("relax.matmul")(x, w)
   out = is_op("relax.add")(matmul, bias)

   pattern = FusionPattern(
       name="cutlass.matmul_bias",
       pattern=out,
       annotation_patterns={"matmul": matmul, "bias": bias},
       check=my_check_function,  # optional validation
   )

Fields:

- ``name``: pattern identifier, typically prefixed with the backend name (e.g.,
  ``"cutlass.matmul_bias"``).
- ``pattern``: a DFPattern describing the subgraph to match. See the
  :ref:`DPL deep dive <relax-dpl>` for the full pattern language.
- ``annotation_patterns``: a mapping of names to sub-patterns within the main pattern. These
  are extracted during matching and made available to the ``check`` function and
  ``attrs_getter``.
- ``check``: an optional ``Callable[[PatternCheckContext], bool]`` that validates whether
  a match should be accepted. Receives the matched expression, annotated sub-expressions,
  variable usages, and binding information.
- ``attrs_getter``: an optional function that extracts attributes (e.g., transpose flags,
  data types) from the matched expressions to annotate the grouped function.

Applying patterns
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tvm.relax.transform import FuseOpsByPattern

   mod = FuseOpsByPattern(
       patterns=[pattern1, pattern2, ...],  # ordered by priority
       bind_constants=True,
       annotate_codegen=False,
   )(mod)

Key parameters:

- ``patterns``: a list of ``FusionPattern`` objects, ordered by priority. Higher-priority
  patterns come first — if a subgraph matches multiple patterns, the first match wins.
- ``bind_constants``: if ``True``, constants used by the matched subgraph are captured inside
  the grouped function.
- ``annotate_codegen``: if ``True``, wraps each composite function with an outer function
  annotated with ``"Codegen"`` and ``"global_symbol"`` attributes for external backend dispatch.
  The ``"Codegen"`` value is derived from the pattern name prefix (e.g., ``"dnnl"`` from
  ``"dnnl.conv2d_relu"``).

PatternCheckContext
~~~~~~~~~~~~~~~~~~~

The ``check`` function receives a ``PatternCheckContext`` with:

- ``matched_expr``: the root expression matched by the pattern.
- ``annotated_expr``: a mapping from annotation pattern names to their matched expressions.
- ``matched_bindings``: variable-to-value bindings within the matched subgraph.
- ``var_usages``: a mapping from variable definitions to all their uses in the function.
- ``value_to_bound_var``: reverse mapping from values to the variables they are bound to.

This context enables sophisticated validation logic, such as checking that an intermediate
result is not used outside the fused group, or verifying data type compatibility.


How Backends Use Fusion
-----------------------

The default backend pipelines (CUDA, ROCm, CPU, etc.) all include ``FuseOps`` + ``FuseTIR``
in their ``legalize_passes`` phase for automatic fusion, as shown in the `Overview`_ above.

For external library dispatch (cuBLAS, CUTLASS, cuDNN, DNNL), ``FuseOpsByPattern`` is used
separately. These are **not** included in the default pipeline — users add them explicitly
when building a custom compilation flow. The typical sequence is:

1. **Pattern-based dispatch** (``FuseOpsByPattern``): identify subgraphs that should be
   offloaded to external libraries. For example, CUTLASS patterns match
   matmul+bias+activation combinations (``python/tvm/relax/backend/cuda/cutlass.py``).
   Functions marked by patterns are annotated with ``Composite`` and optionally ``Codegen``
   attributes. See :ref:`external-library-dispatch` for the full BYOC pipeline.

2. **Automatic fusion** (``FuseOps`` + ``FuseTIR``): remaining operators that were not
   matched by backend patterns are fused automatically based on their pattern kinds.


Source Code Map
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Path
     - Contents
   * - ``src/relax/transform/fuse_ops.cc``
     - FuseOps and FuseOpsByPattern implementation
   * - ``src/relax/analysis/graph_partitioner.h``
     - IndexedForwardGraph, DominatorTree, GraphPartitioner (Union-Find)
   * - ``src/relax/transform/fuse_tir.cc``
     - FuseTIR implementation, SymbolicMatcher
   * - ``include/tvm/relax/op_attr.h``
     - ``OpPatternKind`` enum definition
   * - ``python/tvm/relax/transform/transform.py``
     - Python API: FuseOps, FuseTIR, FuseOpsByPattern, FusionPattern
   * - ``python/tvm/relax/dpl/``
     - Dataflow Pattern Language (DFPattern, is_op, wildcard, etc.)
   * - ``python/tvm/relax/backend/cuda/cutlass.py``
     - Example: CUTLASS fusion patterns
   * - ``python/tvm/relax/backend/cuda/cublas.py``
     - Example: cuBLAS fusion patterns
