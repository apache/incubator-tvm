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

.. _tirx-learning:

Understand TensorIR Abstraction
===============================
TensorIR is the tensor program abstraction in Apache TVM, which is one of the standard
machine learning compilation frameworks. The principal objective of tensor program abstraction
is to depict loops and associated hardware acceleration options, including threading, the
application of specialized hardware instructions, and memory access.

To help our explanations, let us use the following sequence of tensor computations as
a motivating example. Specifically, for two :math:`128 \times 128` matrices ``A`` and ``B``, let us perform the
following two steps of tensor computations.

.. math::

  Y_{i, j} &= \sum_k A_{i, k} \times B_{k, j} \\
  C_{i, j} &= \mathbb{relu}(Y_{i, j}) = \mathbb{max}(Y_{i, j}, 0)


The above computations resemble a typical primitive tensor function commonly seen in neural networks,
a linear layer with relu activation. We use TensorIR to depict the above computations as follows.

Before we invoke TensorIR, let's use native Python codes with NumPy to show the computation:

.. code:: python

    def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
        Y = np.empty((128, 128), dtype="float32")
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[i, j] = 0
                    Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
        for i in range(128):
            for j in range(128):
                C[i, j] = max(Y[i, j], 0)

With the low-level NumPy example in mind, now we are ready to introduce TensorIR. The code block
below shows a TensorIR implementation of ``mm_relu``. The particular code is implemented in a
language called TVMScript, which is a domain-specific dialect embedded in python AST.

.. code:: python

    from tvm.script import tirx as T
    from tvm.script import s_tir as Ts

    @tvm.script.ir_module
    class MyModule:
        @Ts.function
        def mm_relu(A: T.Tensor((128, 128), "float32"),
                    B: T.Tensor((128, 128), "float32"),
                    C: T.Tensor((128, 128), "float32")):
            Y = T.alloc_tensor(ty_args=[T.Tensor((128, 128), "float32")])
            for i, j, k in T.grid(128, 128, 128):
                with Ts.sblock("Y"):
                    vi = Ts.axis.spatial(128, i)
                    vj = Ts.axis.spatial(128, j)
                    vk = Ts.axis.reduce(128, k)
                    with Ts.init():
                        Y[vi, vj] = T.float32(0)
                    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
            for i, j in T.grid(128, 128):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(128, i)
                    vj = Ts.axis.spatial(128, j)
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


Next, let's invest the elements in the above TensorIR program.

Function Parameters and Buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**The function parameters correspond to the same set of parameters on the numpy function.**

.. code:: python

    # TensorIR
    def mm_relu(A: T.Tensor((128, 128), "float32"),
                B: T.Tensor((128, 128), "float32"),
                C: T.Tensor((128, 128), "float32")):
        ...
    # NumPy
    def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
        ...

Here ``A``, ``B``, and ``C`` takes a type named ``T.Tensor``, which with shape
argument ``(128, 128)`` and data type ``float32``. This additional information
helps possible MLC process to generate code that specializes in the shape and data
type.

**Similarly, TensorIR also uses a buffer type in intermediate result allocation.**

.. code:: python

    # TensorIR
    Y = T.alloc_tensor(ty_args=[T.Tensor((128, 128), "float32")])
    # NumPy
    Y = np.empty((128, 128), dtype="float32")

Loop Iterations
~~~~~~~~~~~~~~~
**There are also direct correspondence of loop iterations.**

``T.grid`` is a syntactic sugar in TensorIR for us to write multiple nested iterators.

.. code:: python

    # TensorIR with `T.grid`
    for i, j, k in T.grid(128, 128, 128):
        ...
    # TensorIR with `range`
    for i in range(128):
        for j in range(128):
            for k in range(128):
                ...
    # NumPy
    for i in range(128):
        for j in range(128):
            for k in range(128):
                ...

Computational Block
~~~~~~~~~~~~~~~~~~~
A significant distinction lies in computational statements:
**TensorIR incorporates an additional construct termed** ``Ts.sblock``.

.. code:: python

    # TensorIR
    with Ts.sblock("Y"):
        vi = Ts.axis.spatial(128, i)
        vj = Ts.axis.spatial(128, j)
        vk = Ts.axis.reduce(128, k)
        with Ts.init():
            Y[vi, vj] = T.float32(0)
        Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
    # NumPy
    vi, vj, vk = i, j, k
    if vk == 0:
        Y[vi, vj] = 0
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

A **block** represents a fundamental computation unit within TensorIR. Importantly,
a block encompasses more information than standard NumPy code. It comprises a set of block axes
``(vi, vj, vk)`` and the computations delineated around them.

.. code:: python

    vi = Ts.axis.spatial(128, i)
    vj = Ts.axis.spatial(128, j)
    vk = Ts.axis.reduce(128, k)

The above three lines declare the **key properties** about block axes in the following syntax.

.. code:: python

    [block_axis] = Ts.axis.[axis_type]([axis_range], [mapped_value])

These three lines convey the following details:

- They specify the binding of ``vi``, ``vj``, ``vk`` (in this instance, to ``i``, ``j``, ``k``).
- They declare the original range intended for ``vi``, ``vj``, ``vk``
  (the 128 in ``Ts.axis.spatial(128, i)``).
- They announce the properties of the iterators (spatial, reduce).

Block Axis Properties
~~~~~~~~~~~~~~~~~~~~~
Let's delve deeper into the properties of the block axis. These properties signify the axis's
relationship to the computation in progress. The block comprises three axes ``vi``, ``vj``, and
``vk``, meanwhile the block reads the buffer ``A[vi, vk]``, ``B[vk, vj]`` and writes the buffer
``Y[vi, vj]``. Strictly speaking, the block performs (reduction) updates to Y, which we label
as write for the time being, as we don't require the value of Y from another block.

Significantly, for a fixed value of ``vi`` and ``vj``, the computation block yields a point
value at a spatial location of ``Y`` (``Y[vi, vj]``) that is independent of other locations in ``Y``
(with different ``vi``, ``vj`` values). We can refer to ``vi``, ``vj`` as **spatial axes** since
they directly correspond to the start of a spatial region of buffers that the block writes to.
The axes involved in reduction (``vk``) are designated as **reduce axes**.

Why Extra Information in Block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One crucial observation is that the additional information (block axis range and their properties)
makes the block to be **self-contained** when it comes to the iterations that it is supposed to
carry out independent from the external loop-nest ``i, j, k``.

The block axis information also provides additional properties that help us to validate the correctness of the
external loops that are used to carry out the computation. For example, the above code block will result in an
error because the loop expects an iterator of size 128, but we only bound it to a for loop of size 127.

.. code:: python

    # wrong program due to loop and block iteration mismatch
    for i in range(127):
        with Ts.sblock("C"):
            vi = Ts.axis.spatial(128, i)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            error here due to iterator size mismatch
            ...

Sugars for Block Axes Binding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In situations where each of the block axes is directly mapped to an outer loop iterator,
we can use ``Ts.axis.remap`` to declare the block axis in a single line.

.. code:: python

    # SSR means the properties of each axes are "spatial", "spatial", "reduce"
    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])

which is equivalent to

.. code:: python

    vi = Ts.axis.spatial(range_of_i, i)
    vj = Ts.axis.spatial(range_of_j, j)
    vk = Ts.axis.reduce (range_of_k, k)

So we can also write the programs as follows.

.. code:: python

    @tvm.script.ir_module
    class MyModuleWithAxisRemapSugar:
        @Ts.function
        def mm_relu(A: T.Tensor((128, 128), "float32"),
                    B: T.Tensor((128, 128), "float32"),
                    C: T.Tensor((128, 128), "float32")):
            Y = T.alloc_tensor(ty_args=[T.Tensor((128, 128), "float32")])
            for i, j, k in T.grid(128, 128, 128):
                with Ts.sblock("Y"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    with Ts.init():
                        Y[vi, vj] = T.float32(0)
                    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
            for i, j in T.grid(128, 128):
                with Ts.sblock("C"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
