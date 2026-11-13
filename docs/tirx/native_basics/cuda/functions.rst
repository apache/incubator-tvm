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

Defining a function
===================

A kernel is a ``@Tx.function`` (like ``scale`` in :doc:`first_kernel`), or a
``@Tx.jit`` when it has compile-time parameters (see the last section). This
chapter covers the parameter list — how to declare tensors, what types you can
pass, symbolic shapes, and the ``function`` / ``jit`` distinction.

Declaring tensor parameters
---------------------------

Declare tensor parameters with ``Tx.Tensor`` annotations. The annotation accepts
shape, dtype, layout, offset, scope, and alignment metadata:

.. code-block:: python

    @Tx.function
    def f(A: Tx.Tensor((256,), "float32", align=16), B: Tx.Tensor((256,), "float32")): ...

The parameters are tensors that you index with ``A[i]`` or ``A[i, j]``.
Annotations also support :ref:`symbolic shapes <symbolic-shapes>`.

What the parameter list accepts
-------------------------------

A ``Function`` parameter is one of the following. The third column is what you
pass on the Python side when you call the compiled ``Executable``:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Annotation
     - Is
     - Pass at call time
   * - ``Tx.Tensor((d0, d1), dtype)``
     - a tensor parameter (shape + dtype fixed)
     - a tensor on the right device
   * - ``Tx.handle``
     - an opaque handle
     - a handle value
   * - ``Tx.int32`` / ``Tx.float32`` / …
     - a runtime scalar
     - a Python ``int`` / ``float``
   * - ``Tx.constexpr`` (``@Tx.jit`` only)
     - a compile-time constant
     - supplied to ``.specialize(...)``, **not** at the call

Tensors may be CUDA ``torch`` tensors (zero-copy through TVM FFI tensor
interop) or
``tvm.runtime.tensor(...)``. Arguments are positional and match the parameter
order. For example, a kernel with a scalar parameter::

    @Tx.function
    def scal(A: Tx.Tensor((256,), 'float32'), B: Tx.Tensor((256,), 'float32'), s: Tx.float32):


        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(256,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        B[tx] = A[tx] * s

    exe(a, b, 3.0)        # pass the scalar as a Python float

.. _symbolic-shapes:

Symbolic shapes
---------------

For a size that varies at run time, declare a free symbolic extent with
``Tx.int32()`` and use it in the tensor shape. Its value is **inferred from the
passed tensor** at run time, so a *single compiled kernel* handles any size:

.. code-block:: python

    n = Tx.int32()  # free symbolic extent


    @Tx.function
    def scale_dyn(A: Tx.Tensor((n,), "float32"), B: Tx.Tensor((n,), "float32")):
        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(1,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        for i in range(n):  # loop / launch bounds may use n
            B[i] = A[i] * Tx.float32(2.0)


    exe = tvm.compile(
        tvm.IRModule({"main": scale_dyn}), target=tvm.target.Target("cuda"), tir_pipeline="tirx"
    )
    exe(torch.rand(100, device="cuda"), torch.empty(100, device="cuda"))  # n = 100
    exe(torch.rand(200, device="cuda"), torch.empty(200, device="cuda"))  # n = 200, same kernel

Both tensor annotations share ``n``, so the two shapes are constrained equal;
``n`` is never passed explicitly — it comes from the tensor.

In the generated CUDA, ``n`` is just a runtime kernel argument; the host launcher
reads it from the tensor's shape and passes it, and the loop bound uses it
(boilerplate elided):

.. code-block:: c++

    extern "C" __global__ void
    scale_dyn_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr, int n) {
      for (int i = 0; i < n; ++i) {
        B_ptr[i] = A_ptr[i] * 2.0f;
      }
    }

.. note::

   You passed only two tensors, yet the kernel takes a third argument ``n`` — who
   supplies it? A compiled ``Executable`` has two halves: a **host launcher** and
   the **device kernel** above. When you call ``exe(a, b)``, the host launcher
   unpacks the two tensors, reads ``n`` from ``a``'s shape (``a`` was matched as
   ``(n,)``), checks that ``b`` agrees, computes the launch configuration, and then
   invokes the device kernel — forwarding the data pointers **and** the resolved
   ``n`` as explicit arguments. Nothing passes ``n`` by hand; the host side derives
   it from the tensor metadata. ``tirx.transform.SplitHostDevice`` extracts the
   device function, and the later ``tirx.transform.MakePackedAPI`` pass builds
   the packed host-side argument handling.

You can see it in the IR. **Before** the split, the lowered module is a single
merged function (trimmed):

.. code-block:: python

    n = Tx.int32()  # free symbolic extent


    @Tx.function
    def main(A: Tx.Tensor((n,)), B: Tx.Tensor((n,))):

        with Tx.launch_thread("blockIdx.x", 1), Tx.launch_thread("threadIdx.x", 1):
            for i in range(n):
                B[i] = A[i] * Tx.float32(2.0)

**After** ``SplitHostDevice``, it is two functions — a device kernel that takes
``n`` as a parameter, and a host ``main`` that calls it, forwarding ``n`` (the
trailing ``1, 1`` are the grid/block launch dims):

.. code-block:: python

    @Tx.function  # device
    def scale_dyn_kernel(A_ptr: Tx.handle("float32"), B_ptr: Tx.handle("float32"), n: Tx.int32):
        ...
        for i in range(n):
            B[i] = A[i] * Tx.float32(2.0)


    n = Tx.int32()  # free symbolic extent


    @Tx.function  # host
    def main(A: Tx.Tensor((n,)), B: Tx.Tensor((n,))):

        Tx.call_packed("scale_dyn_kernel", A.data, B.data, n, 1, 1)  # n forwarded

``MakePackedAPI`` then fills in where ``n`` comes from — reading it from the
argument's shape (essentially ``n = a.shape[0]``) — and adds the dtype / shape /
device checks (e.g. asserting ``B.shape[0] == n``)::

    n = Tx.Cast("int32", Tx.abi_field_get(a_shape, 0, 17, "int64"))   # = a.shape[0]

``@Tx.function`` vs ``@Tx.jit``
--------------------------------

- ``@Tx.function`` parses the function immediately into a ``Function``. Sizes are
  whatever you wrote — concrete ints, or runtime-symbolic vars (above).
- ``@Tx.jit`` returns a ``JITFunction`` and **defers** parsing until you call ``.specialize(**constexpr)``:
  parameters annotated ``Tx.constexpr`` are baked in as compile-time constants and
  the result is an ordinary ``Function``. Use it when you want sizes/flags fixed at
  compile time (so the compiler can unroll, statically size shared memory, etc.).
  Referencing a constexpr inside an annotation (e.g. ``Tx.Tensor((N,), ...)``)
  requires ``from __future__ import annotations`` at the top of the file.

.. code-block:: python

    from __future__ import annotations


    @Tx.jit
    def add(
        A: Tx.Tensor((N,), "float32"),
        B: Tx.Tensor((N,), "float32"),
        C: Tx.Tensor((N,), "float32"),
        *,
        N: Tx.constexpr,
    ):
        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(N,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        C[tx] = A[tx] + B[tx]


    kernel = add.specialize(N=256)  # -> a Function with N = 256 baked in

So: a **symbolic shape** is one kernel whose size is resolved at run time; a
**constexpr + jit** produces a specialized kernel per value, resolved at compile
time.

Optional annotations may use ``Tx.Tensor((N,), "float32") | None`` or
``typing.Optional[Tx.Tensor((N,), "float32")]`` (including an imported
``Optional``). The existing ``Tx.Optional(...)`` spelling remains supported.
Use ``from __future__ import annotations`` for these deferred tensor annotations;
they are interpreted as script syntax rather than evaluated as Python unions.
Calling ``kernel.specialize(optional=None)`` skips that parameter's original
annotation and binds Python ``None`` in compile-time guards. The resulting
``Function`` retains the parameter in its original position with ``ir.NoneType``;
the compiled function requires ``None`` in that argument slot. Omitting the
specialization keyword keeps the parameter's ordinary runtime type.

Launch parameters
-----------------

``Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(32,)))``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(32,)))`` starts the authored device region: parameter binding and
shape reads precede it, while the kernel body follows it. A flat call scopes the
remaining statements in the enclosing body; ``with Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(32,))):`` gives
an explicit boundary. Both forms create a ``RegionStmt`` with the
``tirx.device_entry`` op. ``TileDispatch`` lowers this region, resolves scope ids,
and retains its launch operands on a ``device_scope`` region.
``SplitHostDevice`` separates the host configuration from the device body.

CUDA indices and launch configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch geometry is explicit and independent of the indices read by the kernel.
``grid`` counts CTAs, ``block`` counts threads per CTA, and ``cluster`` counts
CTAs per cluster. Each accepts an integer or one to three dimensions; omitted
trailing dimensions are one.

.. code-block:: python

    Tx.device_entry(
        launch=Tx.cuda.LaunchConfig(grid=(GM, GN), block=128),
        kernel_attrs=Tx.cuda.KernelAttributes(min_blocks_per_sm=2),
    )
    bx, by = Tx.cuda.block_idx("x"), Tx.cuda.block_idx("y")
    warp = Tx.cuda.warp_id()
    lane = Tx.cuda.lane_id()
    tid = Tx.cuda.linear_thread_id()

Every assignment above is an ordinary let binding. Tuple assignments produce
separate bindings. Reading an index does not declare or infer launch dimensions,
and unused indices do not remove launch configuration.

The finite index API includes ``block_idx(axis)``, ``thread_idx(axis)``,
``cluster_id(axis)``, ``cluster_cta_id(axis)``, ``grid_dim(axis)``,
``block_dim(axis)``, ``cluster_dim(axis)``, ``linear_thread_id()``, ``warp_id()``,
``lane_id()``, ``warpgroup_id()``, ``warp_in_warpgroup()``,
``thread_in_warpgroup()``, and ``cta_pair_id()``. The axis is ``"x"``, ``"y"``,
or ``"z"``. Linear thread IDs use CUDA's x-major order, including 2D and 3D blocks.
Warp index helpers require a static block size divisible by 32; their shared
full-mask shuffle is emitted at kernel entry, before divergent control flow.

For a cluster launch:

.. code-block:: python

    Tx.device_entry(launch=Tx.cuda.LaunchConfig(
        grid=(NUM_CLUSTERS * 2,), block=128, cluster=(2,),
    ))
    cid = Tx.cuda.cluster_id("x")
    cx = Tx.cuda.cluster_cta_id("x")

Cluster coordinates read the corresponding PTX special registers, including
``%cluster_ctaid.x`` for the CTA's x coordinate. An omitted cluster and an explicit
unit cluster remain distinct. ``preferred_cluster`` requests CUDA's substitute
cluster dimensions; kernels that use it must handle either permitted shape.
Cluster-scope tensor instructions require one static cluster shape. Use explicit
CUDA/PTX instructions when the cluster shape is dynamic or its preferred shape
differs. CTA-, warp-, and thread-scope instructions remain available.

``KernelAttributes`` contains compile-time CUDA kernel attributes: ``min_blocks_per_sm``,
``max_blocks_per_cluster`` (requires ``min_blocks_per_sm``),
``max_registers_per_thread``, and ``required_block_size``. The last attribute fixes
both block and cluster dimensions using CUDA 13's ``__block_size__`` declaration.
``max_registers_per_thread`` conflicts with explicit launch bounds and required
block size. Runtime values belong in ``LaunchConfig``.

Both host backends consume the same validated launch description. The normal
CUDA module and exported ``cuda_host`` code both call the Driver API's
``cuLaunchKernelEx``. The exported code embeds compiled device binaries and
needs only CUDA and tvm-ffi. They share field decoding, attribute encoding, and resource setup.
Dynamic grid sizes, stream handles, event handles, and other launch values are
host-side call operands, not device kernel parameters or expression-valued attrs.

The declaration table in ``python/tvm/backend/cuda/launch/table.py`` generates
the public records, native attribute encoders, and :doc:`launch reference
<../../api/cuda_launch>`. Run ``python python/tvm/backend/cuda/launch/generate.py``
with the repository's pinned Ruff and clang-format versions after editing the
table or shared launch header; pre-commit checks that these files are current.
Historical launch tags are translated only at the packed-call compatibility
boundary. New kernels should use the configuration API above.
