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

Your first kernel
=================

A complete write → compile → run → inspect loop. This kernel scales a vector by 2
with one block of 256 threads.

.. code-block:: python

    import numpy as np
    import tvm
    from tvm.script import tirx as Tx


    @Tx.function
    def scale(A: Tx.Tensor((256,), "float32"), B: Tx.Tensor((256,), "float32")):

        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(256,)))  # everything below runs on the device
        bx = Tx.cuda.block_idx("x")  # 1 block  (blockIdx)
        tx = Tx.cuda.thread_idx("x")  # 256 threads per block (threadIdx)

        B[tx] = A[tx] * Tx.float32(2.0)


    # compile for CUDA through the TIRx pipeline -> an Executable
    exe = tvm.compile(
        tvm.IRModule({"main": scale}), target=tvm.target.Target("cuda"), tir_pipeline="tirx"
    )

    dev = tvm.cuda(0)
    a = tvm.runtime.tensor(np.random.rand(256).astype("float32"), device=dev)
    b = tvm.runtime.tensor(np.zeros(256, "float32"), device=dev)
    exe(a, b)  # run
    print(exe.mod.imports[0].inspect_source())  # the generated CUDA C

What the surrounding calls do:

- ``tvm.IRModule({"main": scale})`` wraps the ``Function`` into an *IRModule* — a
  named collection of functions; ``"main"`` is the entry point the compiler builds
  and the symbol you call.
- ``tvm.compile(mod, target=..., tir_pipeline="tirx")`` returns an **Executable**:
  the compiled host launcher plus device code, bundled together. You call it like
  a function (``exe(a, b)``); arguments are positional and match the kernel's
  parameter order.
- ``tvm.cuda(0)`` is a handle to CUDA device 0; ``tvm.runtime.tensor(arr,
  device=dev)`` places data on that device as a TVM tensor (an ``NDArray``).
- ``exe.mod`` is the underlying runtime ``Module``; ``exe.mod.imports[0]`` is the
  imported device (CUDA) module, and its ``.inspect_source()`` returns the
  generated CUDA C.

For this kernel, that last ``print`` produces (boilerplate elided):

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(256)
    scale_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tx = ((int)threadIdx.x);
      B_ptr[tx] = A_ptr[tx] * 2.0f;
    }

Every thread writes one element — a direct map from ``B[tx] = A[tx] * 2.0`` to the
generated indexing.

.. note::

   The compiled ``Executable`` also accepts CUDA ``torch`` tensors **directly**
   through TVM FFI tensor interop, without an explicit conversion step:

   .. code-block:: python

       import torch
       a = torch.rand(256, device="cuda")
       b = torch.empty(256, device="cuda")
       exe(a, b)
       torch.testing.assert_close(b, a * 2)

The following chapters expand each piece: :doc:`functions`, :doc:`tensors`,
:doc:`control_flow`, :doc:`threads_sync`, and :doc:`compiling`.
