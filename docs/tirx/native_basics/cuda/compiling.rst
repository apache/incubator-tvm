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

Compiling and inspecting
========================

CUDA compilation accepts a lightweight ``backend_config`` mapping. The same
nested structure is accepted by the Target's ``backend_config`` attribute and
``T.device_entry``. Pass the configured Target to ``tvm.compile`` or
``txl.Kernel.compile``. ``tvm.backend.cuda.BackendConfig`` is an optional
``TypedDict`` helper, also available as ``T.cuda.BackendConfig`` and
``txl.cuda.BackendConfig``:

.. code-block:: python

    from tvm.backend.cuda import BackendConfig

    @T.function
    def pipeline(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            backend_config={"cuda": {"nvrtc": ["--use_fast_math", "--ftz=false"]}},
        ):
            x = T.cuda.thread_idx("x")
            B[x] = A[x] * T.float32(0.5)
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            backend_config={"cuda": {
                "compiler": "nvcc",
                "nvcc": ["--use_fast_math", "--generate-line-info"],
            }},
        ):
            y = T.cuda.thread_idx("x")
            A[y] = B[y] + T.float32(1)

    target = tvm.target.Target({
        "kind": "cuda", "arch": "sm_100a",
        "backend_config": {"cuda": BackendConfig(compiler="nvrtc")},
    })
    exe = tvm.compile(pipeline, target=target)

Each entry resolves its configuration before architecture-sensitive lowering.
Entries with different resolved targets or options compile as separate CUDA
modules; their shared device helpers are copied into each group. The existing
host module imports these device modules and launches them in program order.
Kernels must be compatible with the device on which they execute.

Defaults and overrides
----------------------

Precedence, from lowest to highest, is backend defaults, Target/tag settings,
then device-entry overrides. ``None`` and ``{}`` add no
overrides. Toolchain lists replace inherited lists in full, and ``[]`` clears a
list. To keep fast math while changing FTZ, include both arguments as above.
Configuration boundaries snapshot dictionaries and lists, so later mutation of
the input cannot change an already constructed entry or a factory cache key.

Target tags use the existing target registry and carry defaults in Target attrs:

.. code-block:: python

    tvm.target.tag.register_tag("local/blackwell", {
        "kind": "cuda", "arch": "sm_100a",
        "backend_config": {"cuda": {
            "compiler": "nvcc", "nvcc": ["--use_fast_math", "--generate-line-info"],
        }},
    })
    exe = tvm.compile(pipeline, target="local/blackwell")

Both ``Target("local/blackwell")`` and ``Target({"tag": "local/blackwell"})``
preserve these defaults in their expanded attributes. There is no separate
backend tag registry. An entry can override the architecture supplied by its Target.

Online builds can detect the GPU architecture. Offline builds require an
explicit architecture in the Target or every device entry.
A generic ``Target("cuda")`` may remain without an architecture for backend
discovery and IR construction; compilation validates it before lowering.
Architecture-dependent Python factories resolve and record their architecture
before tracing. Other backend defaults remain overridable at compilation.

The default compiler is NVRTC, with fast math enabled and ptxas register usage
level 10. NVRTC produces cubin by default, NVCC produces fatbin, and NVSHMEM
requires cubin. Native options go into ``nvcc``, ``nvrtc``, or ``ptxas`` as
individual argv strings; the compiler validates their values and availability.
TVM reserves only routing, architecture, and output controls that it manages.
Adding a compiler flag does not require adding a Python field or updating TVM.

``LaunchConfig`` supplies runtime grid, block, cluster, and stream settings.
``KernelAttributes`` describes CUDA kernel declaration attributes.
``backend_config`` controls device compilation. PassContext still controls TVM
passes independently; there is no wrapper combining these APIs. Only CUDA
backend configuration is implemented currently; unsupported backend keys raise
an explicit error.

Artifacts and standalone hosts
------------------------------

CUDA modules retain the resolved configuration as one opaque JSON string,
available through ``module.inspect_source("backend_config")``. Serialization
keeps the existing format, function map, and code fields, then appends this string.
It does not persist compiler logs or the complete source map. Source fallback
artifacts retain source and configuration and compile on loading with a CUDA
runtime. Replay uses the saved defaults independently of the loader's Target scope. Producing a source fallback does not require NVCC or NVRTC.
Old binary artifacts remain loadable; old source artifacts without configuration
use the default compiler settings when compiled.

For ``target=Target({"kind": "cuda", "arch": "sm_100a"}, host="cuda_host")``,
``tvm.backend.cuda.export_cuda_host(exe.mod)`` returns ordinary C++ host source
with embedded independently compiled device binaries. Compile it with a C++
compiler and CUDA headers, and link with tvm-ffi, cudart and the CUDA driver.
The exported library requires no TVM installation or device compiler at runtime.
Compiling the host does not recompile kernels.

Migration
---------

Replace the former compilation dataclass with the nested mapping above. Convert
math, debugging, include, and assembler options to native argument strings.
Replace ``@txl.kernel(arch=...)`` and ``tirx.cuda_arch`` with ``Target.arch`` or
an entry's ``backend_config={"cuda": {"arch": ...}}``.

Compiler/math/ptxas environment policies remain removed. CPU benchmark
preparation receives an explicit architecture through ``backend_config`` and
defaults to NVCC. The runner, benchmark, and test CLIs accept
``--backend-config '{"cuda":{"arch":"sm_100a","compiler":"nvcc"}}'``.
Factory caches use canonical configuration snapshots; dictionary key order is
ignored while argument order remains significant.

See :doc:`../../api/cuda_compile` for the complete configuration API.

Inspecting the result
---------------------

Read the IR with ``.show()`` / ``.script()``, and read the generated CUDA from the
compiled module.

.. code-block:: python

    pipeline.show()                       # pretty-print the TIRx (TVMScript)
    print(pipeline.script())              # ... the same, as a string

    # the generated CUDA C source, from the compiled Executable:
    print(exe.mod.imports[0].inspect_source())

``Tx.hint("message")`` (statement or ``with`` block) attaches structured hints
that survive a script round-trip.

From simple to complex
----------------------

A natural native progression, each rung adding one capability:

#. **Elementwise** — ``device_entry`` + ``thread_id`` + a guarded store (the first
   kernel).
#. **Shared-memory reduction** — stage into ``Tx.alloc_shared``, then a
   ``cta_sync``-separated tree (shown in full below). Adds shared memory and a
   block barrier.
#. **Warp / block reduction** — ``Tx.gpu_warp_shuffle_xor`` or ``Tx.cuda.cta_sum``
   to combine partial results across lanes/warps (the warp all-reduce in
   :doc:`threads_sync`).
#. **Async pipeline** — ``Tx.ptx.cp.async_`` (or TMA ``cp.async.bulk.tensor``) with
   ``Tx.ptx.mbarrier.*`` / ``Tx.cuda.mbarrier_wait`` to overlap loads with compute.

Rung 2 in full — a 256-element block sum via a shared-memory tree reduction
(shared tensor, ``cta_sync``, a ``while`` loop, and a thread predicate):

.. code-block:: python

    @Tx.function
    def block_sum(A: Tx.Tensor((256,), "float32"), out: Tx.Tensor((1,), "float32")):

        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(256,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")

        sm = Tx.alloc_shared((256,), "float32")
        sm[tx] = A[tx]
        Tx.cuda.cta_sync()

        s = Tx.alloc_local((1,), "int32")
        s[0] = 128
        while s[0] >= 1:
            if tx < s[0]:
                sm[tx] += sm[tx + s[0]]
            Tx.cuda.cta_sync()
            s[0] = s[0] // 2

        if tx == 0:
            out[0] = sm[0]


    exe = tvm.compile(
        tvm.IRModule({"main": block_sum}), target=tvm.target.Target("cuda"), tir_pipeline="tirx"
    )
    a = torch.arange(256, device="cuda", dtype=torch.float32)
    out = torch.zeros(1, device="cuda")
    exe(a, out)  # out[0] == 32640.0

The full tile-level GEMM/attention ladder (sync → TMA → warp specialization →
2-CTA cluster) is built on top of these and the dispatchable tile ops in
:doc:`../../tile_ops`.

Next steps
----------

- :doc:`../../layout` — how tensors map to physical resources (``TileLayout``).
- :doc:`../../tile_ops` — the dispatchable ops these native idioms lower to.
