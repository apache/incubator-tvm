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

.. _relax-abstraction:

Graph Abstraction for ML Models
-------------------------------
Graph abstraction is a key technique used in machine learning (ML) compilers
to represent and reason about the structure and data flow of ML models. By
abstracting the model into a graph representation, the compiler can perform
various optimizations to improve performance and efficiency. This tutorial will
cover the basics of graph abstraction, its key elements of Relax IR, and how it enables optimization in ML compilers.

What is Graph Abstraction?
~~~~~~~~~~~~~~~~~~~~~~~~~~
Graph abstraction is the process of representing an ML model as a directed graph,
where the nodes represent computational operations (e.g., matrix multiplication,
convolution) and the edges represent the flow of data between these operations.
This abstraction allows the compiler to analyze the dependencies and
relationships between different parts of the model.

.. code:: python

    from tvm.relax import script as R

    @R.function
    def main(
        x: R.Tensor((1, 784), dtype="float32"),
        weight: R.Tensor((784, 256), dtype="float32"),
        bias: R.Tensor((256,), dtype="float32"),
    ) -> R.Tensor((1, 256), dtype="float32"):
        with R.dataflow():
            lv0 = R.matmul(x, weight)
            lv1 = R.add(lv0, bias)
            gv = R.nn.relu(lv1)
            R.output(gv)
        return gv

Key Features of Relax
~~~~~~~~~~~~~~~~~~~~~
Relax, the graph representation utilized in Apache TVM,
facilitates end-to-end optimization of ML models through several crucial
features:

- **First-class symbolic shape**: Relax employs symbolic shapes to represent
  tensor dimensions, enabling global tracking of dynamic shape relationships
  across tensor operators and function calls.

- **Multi-level abstractions**: Relax supports cross-level abstractions, from
  high-level neural network layers to low-level tensor operations, enabling
  optimizations that span different hierarchies within the model.

- **Composable transformations**: Relax offers a framework for composable
  transformations that can be selectively applied to different model components.
  This includes capabilities such as partial lowering and partial specialization,
  providing flexible customization and optimization options.

These features collectively empower Relax to offer a powerful and adaptable approach
to ML model optimization within the Apache TVM ecosystem.
