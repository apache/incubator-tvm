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

TIRx lowering pipeline
======================

``tvm.compile(mod, target, tir_pipeline="tirx")`` runs an authored TIRx module
through the **tirx pipeline** — an ordered sequence of IR passes that turns the
high-level constructs you write (tile ops, ``TileLayout``-typed tensors,
CUDA index calls) into split **host** + **device** functions, which the CUDA
backend then renders to source. The pipeline is defined in
``python/tvm/tirx/compilation_pipeline.py`` (``tirx_pipeline``); this page walks the
passes in order.

Where it sits
-------------

``tvm.compile`` first binds the target, runs the **tirx pipeline** (the module-level
passes below), then applies **finalization** passes separately to the host and
device functions, and finally hands each device function to the CUDA code
generator:

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                          │
                                          └──────────▶  device func ──device finalize──▶  CUDA

The passes
----------

S-TIR represents scheduled thread placement with ordinary ``T.parallel`` loops
and a visible ``thread_binding`` annotation. S-TIR normalization removes
unannotated extent-one loops, and ``LowerThreadBinding`` converts thread-bound
loops to explicit ``T.launch_thread`` regions before generic TIRx transforms.
TIRx source can construct these regions directly. The generic pipeline uses
``StmtSimplify`` for arithmetic simplification and ``UnrollLoop`` for inherited
loop unroll policies.

The ``tirx_pipeline`` module pass applies this sequence (a few passes are gated by
``PassContext`` config):

.. list-table::
   :header-rows: 1
   :widths: 6 32 62

   * - #
     - Pass
     - What it does
   * - 1
     - ``TileDispatch`` and ``LowerTensorLayouts``
     - the core lowering — see `Tile dispatch and tensor layouts`_ below
   * - 2
     - ``StmtSimplify``
     - statement-level arithmetic simplification (the sym analyzer)
   * - 3
     - ``FlattenTensor``
     - flattens multi-dimensional ``TensorLoad`` / ``TensorStore`` to 1-D
   * - 4
     - ``BF16ComputeLegalize``
     - rewrites ``bfloat16`` compute to a legal (f32-up-cast) form
   * - 5
     - ``NarrowDataType(32)``
     - narrows index/loop ``PrimExpr`` dtypes to 32-bit where provably safe
   * - 6
     - ``VectorizeLoop``
     - turns ``Tx.vectorized`` loops into vector ops (skipped if
       ``tir.disable_vectorize``)
   * - 7
     - ``UnrollLoop``
     - unrolls loops marked ``Tx.unroll`` (and small constant loops)
   * - 8
     - ``StmtSimplify``
     - simplify again, now that vectorize/unroll exposed constants
   * - 9
     - ``CommonSubexprElim``
     - hoists repeated subexpressions into temporaries (skipped if
       ``tir.disable_cse_tir``)
   * - 10
     - ``FP8ComputeLegalize``
     - rewrites ``float8`` compute to a legal form
   * - 11
     - ``VerifyMemory``
     - checks no host-side code directly dereferences device memory (a safety gate)
   * - 12
     - ``AnnotateEntryFunc``
     - marks the single Function as the module entry point
   * - 13
     - ``SplitHostDevice``
     - extracts target-annotated device regions into **device** functions and
       leaves launch calls in the **host** function; the regions originate from
       the explicit ``LaunchConfig`` attached to ``Tx.device_entry``
   * - 14
     - ``LowerIket``
     - lowers CUDA IKET instrumentation after host/device splitting
   * - 15
     - ``MakePackedAPI``
     - rewrites the host function to the packed-func ABI (the launcher TVM calls)
   * - 16
     - ``FP8StorageLegalize``
     - legalizes ``float8`` storage (packing into supported container types)
   * - 17
     - ``BF16StorageLegalize``
     - legalizes ``bfloat16`` storage

**Finalization** then runs per function kind:

- **host**: ``LowerTVMBuiltin`` (lower ``tvm_*`` builtins), ``LowerIntrin``
  (target-specific intrinsics)
- **device**: ``StmtSimplify``,
  ``LowerIntrin``

Tile dispatch and tensor layouts
--------------------------------

The generic pipeline explicitly composes two independently usable passes:

.. code-block:: text

    Sequential([ TileDispatch(), LowerTensorLayouts() ])

- **``TileDispatch``** replaces every tensor ``Evaluate(Call)`` with the body emitted by its
  instruction lowerer, including delayed mathematical composites as described in
  :doc:`tile_dispatch`. In the same pass it converts ``device_entry`` to a
  ``device_scope`` region retaining launch values as operands, and lowers the
  finite CUDA index calls inside ordinary ``Bind`` statements.
- **``LowerTensorLayouts``** then runs the ``LayoutApplier``: it resolves every
  ``TileLayout``-typed tensor access into concrete physical address arithmetic
  (``addr = data + layout.apply(coord, shape=shape)["m"]``, in elements),
  flattens the tensors, and accounts for the layout's memory origin exactly once
  in the resulting view or its access indices.

A backend can pair ``TileDispatch`` with its own layout lowering, such as
Trainium's ``LowerTrainiumLayout``. The generic pipeline inserts ``PrintIR``
between dispatch and layout lowering when ``TVM_TIRX_PRINT_AFTER_DISPATCH_OPS``
is set.

After both passes the module remains a ``tvm.tirx.Function``, but contains no
tile ops or ``TileLayout`` indirection, and CUDA indices read hardware coordinates.  Later TIRx passes lower the remaining opaque constructs and
the target code generators consume ``tirx::Function`` directly; there is no
conversion to the separate ``tvm.tir`` object model.

A worked example
----------------

Take a one-line scale kernel:

.. code-block:: python

    @Tx.function
    def scale(A: Tx.Tensor((256,), "float32"), B: Tx.Tensor((256,), "float32")):

        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(256,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        B[tx] = A[tx] * Tx.float32(2.0)

**After both passes** the layout is applied and the device region retains its
launch configuration independently of the index bindings:

.. code-block:: python

    with Tx.region("tirx.device_scope", [1, 1, 1, 256, 1, 1], attrs={
        "cuda.launch_fields": ["grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z"],
        "cuda.kernel_attrs": {},
    }):
        bx: Tx.let = 0
        tx: Tx.let = Tx.cuda.thread_idx("x")
        B_1[tx] = A_1[tx] * Tx.float32(2.0)

**After ``SplitHostDevice`` and the later ``MakePackedAPI`` pass** the one function
has become two —
a host launcher and a device kernel:

.. code-block:: python

    @I.ir_module
    class Module:
        def main(...):          # host: packed-API launcher (computes the grid/block, launches)
            ...
        def scale_kernel(...):  # device: the __global__ body, run on the GPU

The CUDA backend then renders ``scale_kernel`` to the ``__global__`` function
(``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f``).

Reproduce it yourself
---------------------

You can run any prefix of the pipeline by hand to inspect a stage — this is how the
IR snippets across these docs were produced:

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda")
    mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
    mod = TT.TileDispatch()(mod)         # tile ops dispatched
    mod = TT.LowerTensorLayouts()(mod)   # layouts applied
    print(mod.script())               # inspect the lowered TIRx IR

Or compile the whole module and read the generated CUDA:

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    print(exe.mod.imports[0].inspect_source())
