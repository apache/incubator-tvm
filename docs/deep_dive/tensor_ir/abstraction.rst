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

.. _tir-abstraction:

Tensor Program Abstraction
--------------------------
Before we dive into the details of TensorIR, let's first introduce what is a primitive tensor
function. Primitive tensor functions are functions that correspond to a single "unit" of
computational operation. For example, a convolution operation can be a primitive tensor function,
and a fused convolution + relu operation can also be a primitive tensor function.
Usually, a typical abstraction for primitive tensor function implementation contains the following
elements: multi-dimensional buffers, loop nests that drive the tensor computations, and finally,
the compute statements themselves.

.. code:: python

    from tvm.script import tirx as T
    from tvm.script import s_tir as Ts

    @Ts.function
    def main(
        A: T.Tensor((128,), "float32"),
        B: T.Tensor((128,), "float32"),
        C: T.Tensor((128,), "float32"),
    ) -> None:
        for i in range(128):
            with Ts.sblock("C"):
                vi = Ts.axis.spatial(128, i)
                C[vi] = A[vi] + B[vi]

Key Elements of Tensor Programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The demonstrated primitive tensor function calculates the element-wise sum of two vectors.
The function:

- Accepts three **multi-dimensional buffers** as parameters, and generates one **multi-dimensional
  buffer** as output.
- Incorporates a solitary **loop nest** ``i`` that facilitates the computation.
- Features a singular **compute statement** that calculates the element-wise sum of the two
  vectors.

Extra Structure in TensorIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Crucially, we are unable to execute arbitrary transformations on the program, as certain
computations rely on the loop's sequence. Fortunately, the majority of primitive tensor
functions we focus on possess favorable properties, such as independence among loop iterations.
For instance, the aforementioned program includes block and iteration annotations:

- The **block annotation** ``with Ts.sblock("C")`` signifies that the block is the fundamental
  computation unit designated for scheduling. A block may encompass a single computation
  statement, multiple computation statements with loops, or opaque intrinsics such as Tensor
  Core instructions.
- The **iteration annotation** ``Ts.axis.spatial``, indicating that variable ``vi`` is mapped
  to ``i``, and all iterations are independent.

While this information isn't crucial for *executing* the specific program, it proves useful when
transforming the program. Consequently, we can confidently parallelize or reorder loops associated
with ``vi``, provided we traverse all the index elements from 0 to 128.
