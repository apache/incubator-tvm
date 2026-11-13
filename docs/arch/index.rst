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

Design and Architecture
=======================

This document is intended for developers who want to understand the architecture of Apache TVM and/or actively develop on the project.
This page is organized as follows:

- The `Overall Flow`_ gives an overview of the steps that TVM takes to turn a high level description of a model into a deployable module.
  To get started, please read this section first.
- Brief introduction to the key components of the TVM stack. Feel free to also check out the :ref:`TensorIR Deep Dive <tensor-ir-deep-dive>`
  and :ref:`Relax Deep Dive <relax-deep-dive>` for more details about the two major components in the TVM stack.

This guide provides a few complementary views of the architecture.
First, we review a single end-to-end compilation flow and discuss the key data structures and the transformations.
This runtime-based view focuses on the interactions of each components when running the compiler.
Then we will review the logical modules of the codebase and their relationship. This part provides a static overarching view of the design.

Overall Flow
------------

In this guide, we will study an example compilation flow in the compiler. The figure below shows the flow. At a high-level, it contains several steps:

- **Model Creation**: Create the IRModule to be optimized and compiled, which contains a collection of functions that internally represent the model.
  Users can manually construct IRModule via NNModule, TVMScript, or import a pre-trained model from Relax frontend.
- **Transformation**: The compiler transforms an IRModule to another functionally equivalent or approximately
  equivalent(e.g. in the case of quantization) IRModule. Many of the transformations are target (backend) independent.
  We also allow target to affect the configuration of the transformation pipeline.
- **Target Translation**: The compiler translates(codegen) the IRModule to an executable format specified by the target.
  The target translation result is encapsulated as a `tvm_ffi.Module` that can be exported, loaded, and executed on the target runtime environment.
- **Runtime Execution**: the user loads back a `tvm_ffi.Module` and runs the compiled functions in the supported runtime environment.


.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/design/tvm_overall_flow.svg
   :align: center
   :width: 80%


Key data structures
~~~~~~~~~~~~~~~~~~~

One of the best ways to design and understand a complex system is to identify the key data structures and APIs that
manipulate (transform) these data structures. Once we identified the key data structures, we can then breakdown a system into logical
components that either define a collection of key data structures or transformations among the data structures.

**IRModule** is the primary data structure used across the entire stack. An IRModule (intermediate representation module)
contains a collection of functions. Currently, we support two primary variants of functions.

- **relax::Function** is a high-level functional program representation. A relax.Function represents high-level graph structure,
  usually corresponds to an end-to-end model or a sub-graph of the overall model. You can view a relax.Function as a computational
  graph with additional support for control-flow, and complex data structures.
- **tirx::Function** is a low-level program representation that contains elements including loop-nest choices, multi-dimensional load/store,
  threading, and vector/tensor instructions. It is usually used to represent an operator program that executes a (possibly-fused) layer in a model.

During the compilation and transformation, all relax operators are lowered to ``tirx::Function`` or ``TVM PackedFunc``, which can be executed directly
on the target device, while the calls to relax operators are lowered to calls to low-level functions (e.g. ``R.call_tir`` or ``R.call_dps_packed``).

Transformations
~~~~~~~~~~~~~~~

Now that we have covered the key data structures, let us talk about the transformations. Each transformation could serve one of the following purposes:

- optimization: transform a program to an equivalent, possibly more optimized version.
- lowering: transform a program to a lower-level representation that is closer to the target.

relax transformations
^^^^^^^^^^^^^^^^^^^^^
relax transformations contain a collection of passes that apply to relax functions. The optimizations include common graph-level
optimizations such as constant folding and dead-code elimination for operators, and backend-specific optimizations such as library dispatch.

TensorIR transformations
^^^^^^^^^^^^^^^^^^^^^^^^

- **TensorIR schedule**: TensorIR schedules are designed to optimize the TensorIR functions for a specific target, with user-guided instructions and control how the target code is generated.
  For CPU targets, a TensorIR Function can generate valid code and execute on the target device without schedule but with very-low performance. However, for GPU targets, the schedule is essential
  for generating valid code with thread bindings. For more details, please refer to the :ref:`TensorIR Transformation <tirx-transform>` section. Additionally, we provides ``MetaSchedule`` to
  automate the search of TensorIR schedule.
- **Lowering Passes**: These passes usually perform after the schedule is applied, transforming a TensorIR Function into another functionally equivalent Function, but closer to the
  target-specific representation. For example, there are passes to flatten multi-dimensional access to one-dimensional pointer access, to expand the intrinsics into target-specific ones,
  and to decorate the function entry to meet the runtime calling convention.

Many low-level optimizations can be handled in the target phase by the LLVM,
CUDA C, and other target compilers. As a result, we leave low-level
optimizations such as register allocation to the downstream compilers and only
focus on optimizations that are not covered by them.

cross-level transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Apache TVM enables cross-level optimization of end-to-end models. As the IRModule includes both Relax and TensorIR functions, the cross-level transformations are designed to mutate
the IRModule by applying different transformations to these two types of functions.

For example, ``relax.LegalizeOps`` pass mutates the IRModule by lowering relax operators, adding corresponding TensorIR Function into the IRModule, and replacing the relax operators
with calls to the lowered TensorIR Function. Another example is the operator fusion pipeline
(``relax.FuseOps`` + ``relax.FuseTIR``), which fuses multiple consecutive tensor operations into a
single kernel. See :ref:`fusion-arch` for a detailed explanation of the fusion algorithm, operator
pattern classification, and pattern-based fusion for external backends.

.. toctree::
   :maxdepth: 1

   fusion

Target Translation
~~~~~~~~~~~~~~~~~~

The target translation phase transforms an IRModule to the corresponding target executable format.
For backends such as x86 and ARM, we use the LLVM IRBuilder to build in-memory LLVM IR.
We can also generate source-level languages such as CUDA C and OpenCL.
Finally, we support direct translations of a Relax function (sub-graph) to specific targets via external code generators.
See :ref:`codegen-arch` for how TIR functions are compiled to native code through the LLVM and
Source codegen families.
See :ref:`external-library-dispatch` for the full BYOC (Bring Your Own Codegen) pipeline that
offloads operator subgraphs to vendor libraries like cuBLAS, CUTLASS, and cuDNN.
It is important that the final code generation phase is as lightweight as possible. Vast majority of transformations
and lowering should be performed before the target translation phase.

.. toctree::
   :maxdepth: 1

   codegen
   external_library_dispatch

We also provide a Target structure to specify the compilation target.
The transformations before the target translation phase can also be affected by the target — for example,
a target's vector length would change the vectorization behavior.


Runtime Execution
~~~~~~~~~~~~~~~~~

The main goal of TVM's runtime is to provide a minimal API for loading and executing the compiled artifact in a language of their choice, including Python, C++, Rust, Go, Java, and JavaScript. The code snippet below shows such an example in Python:

.. code-block:: python

    import tvm
    import tvm_ffi
    # Example runtime execution program in python, with type annotated
    mod: tvm_ffi.Module = tvm.runtime.load_module("compiled_artifact.so")
    arr: tvm.runtime.Tensor = tvm.runtime.tensor([1, 2, 3], device=tvm.cuda(0))
    fun: tvm_ffi.Function = mod["addone"]
    fun(arr)
    print(arr.numpy())


:py:class:`tvm_ffi.Module` encapsulates the result of compilation. A tvm_ffi.Module contains a GetFunction method to obtain :py:class:`tvm_ffi.Function` instances by name.

:py:class:`tvm_ffi.Function` is a type-erased function interface for both the generated functions. A tvm_ffi.Function can take arguments and return values with the
following types: POD types(int, float), string, tvm_ffi.Function, tvm_ffi.Module, runtime.Tensor, and other sub-classes of tvm_ffi.Object.

:py:class:`tvm_ffi.Module` and :py:class:`tvm_ffi.Function` are powerful mechanisms to modularize the runtime. For example, to get the above `addone` function on CUDA, we can use LLVM to generate the host-side code to compute the launching parameters(e.g. size of the thread groups) and then call into another tvm_ffi.Function from a CUDAModule that is backed by the CUDA driver API. The same mechanism can be used for OpenCL kernels.

The above example only deals with a simple `addone` function. The code snippet below gives an example of an end-to-end model execution using the Relax Virtual Machine, which is built on the same tvm_ffi.Module and tvm_ffi.Function interface:

.. code-block:: python

   import tvm
   import tvm_ffi
   from tvm import relax
   # Load the compiled artifact
   mod: tvm_ffi.Module = tvm.runtime.load_module("resnet18.so")
   # Create a VM instance on cuda(0)
   vm = relax.VirtualMachine(mod, tvm.cuda(0))
   data: tvm.runtime.Tensor = get_input_data()
   # Run the model — vm["main"] returns a PackedFunc
   result = vm["main"](data).numpy()

The main take away is that tvm_ffi.Module and runtime.PackedFunc are sufficient to encapsulate both operator level programs (such as addone), as well as the end-to-end models.

Summary and Discussions
~~~~~~~~~~~~~~~~~~~~~~~

In summary, the key data structures in the compilation flows are:

- IRModule: contains relax.Function and tirx.Function
- tvm_ffi.Module: contains runtime.PackedFunc

Most parts of the compilation are transformations among the key data structures.

- relax/transform and tirx/transform are deterministic rule-based transformations
- meta-schedule contains the search-based transformations

Finally, the compilation flow example is only a typical use-case of the TVM stack.
We expose these key data structures and transformations to python and C++ APIs. As a result, you can use TVM just like the way you use numpy,
except that the data structure of interest changes from the numpy.ndarray to tvm.IRModule. Here are some example use-cases:

- Directly construct IRModule using the python API.
- Compose a custom set of transformations(e.g. customize quantization).
- Manipulate the IR directly using TVM's python API.


tvm/support
-----------
The support module contains the most common utilities for the infrastructure, such as generic arena allocator, socket, and logging.


tvm/runtime
-----------

The runtime serves as the foundation of the TVM stack. It provides the mechanism to load and execute compiled artifacts.
The runtime defines a stable standard set of C APIs to interface with frontend languages such as Python and Rust.

`runtime::Object` is one of the primary data structures in TVM runtime besides the `ffi::Function`.
It is a reference-counted base class with a type index to support runtime type checking and downcasting.
The object system allows the developer to introduce new data structures to the runtime, such as Array, Map, and new IR data structures.

Besides deployment use-cases, the compiler itself also makes heavy use of TVM's runtime mechanism.
All of the IR data structures are subclasses of `runtime::Object`, as a result, they can be directly accessed and manipulated from the Python frontend.
We use the PackedFunc mechanism to expose various APIs to the frontend.

Runtime support for different hardware backends are defined in subdirectories of runtime(e.g. runtime/opencl).
These hardware-specific runtime modules define APIs for device memory allocation and device function serialization.

`runtime/rpc` implements an RPC support for PackedFunc. We can use the RPC mechanism to send a cross-compiled library to a remote
device and benchmark the execution performance. The rpc infrastructure enables data collection from a wide range of hardware backends
for learning-based optimizations.

.. toctree::
   :maxdepth: 1

   runtime

.. toctree::
   :maxdepth: 1

   introduction_to_module_serialization

Relax Virtual Machine
~~~~~~~~~~~~~~~~~~~~~

Relax defines *what* to compute — it is a graph-level IR that describes the operators and dataflow
of a model. The Relax Virtual Machine (VM) handles *how* to run it — it is the runtime component
that executes the compiled result. The VM uses a register-based interpreter with only four opcodes
(``Call``, ``Ret``, ``Goto``, ``If``) and performs no mathematical computation itself — it
orchestrates control flow while dispatching actual work to compiled TIR kernels or external
libraries.

See :ref:`relax-vm-arch` for the full architecture documentation, including the compilation
pipeline, instruction set details, execution model, and Python interface.

.. toctree::
   :maxdepth: 1

   relax_vm

Disco: Distributed Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~

Disco is TVM's distributed runtime for executing models across multiple devices. When a model is
too large to fit on a single GPU, the ``relax.distributed`` module annotates how tensors should be
partitioned and placed across a mesh of devices at compile time. Disco then takes over at runtime:
it manages a group of workers, dispatches the compiled program to all of them simultaneously, and
coordinates inter-device communication through collective operations such as allreduce, allgather,
broadcast, and scatter.

The central abstraction is the ``Session``, which owns the workers and exposes a SPMD-style
programming interface. Every object that lives on workers is represented by a ``DRef`` — a
distributed reference that maps to a concrete value on each worker. When the controller invokes a
``DPackedFunc`` through the session, all workers execute the same PackedFunc call synchronously, each
operating on its own local shard. Compiled VM modules can be loaded into a session as ``DModule``
objects and called in the same fashion. The session also provides collective primitives backed by
NCCL or RCCL, so that workers can exchange partial results without routing data through the
controller.

Three session backends cover different deployment topologies. ``ThreadedSession`` spawns workers as
threads within a single process — this is the most common choice for multi-GPU inference on a
single machine. ``ProcessSession`` launches workers as separate OS processes connected by pipes,
providing stronger isolation. ``SocketSession`` extends the model to multi-node clusters by
connecting workers across machines via TCP sockets.

tvm/node
--------
The node module adds additional features on top of the `runtime::Object` for IR data structures.
The main features include reflection, serialization, structural equivalence, and hashing.

Thanks to the node module, we can directly access any field of the TVM's IRNode by their name in Python.

.. code-block:: python

    x = tvm.tirx.Var("x", "int32")
    y = tvm.tirx.Add(x, x)
    # a and b are fields of a tirx.Add node
    # we can directly use the field name to access the IR structures
    assert y.a == x

We can also serialize arbitrary IR node into a JSON format, and load them back.
The ability to save/store, and inspect an IR node provides a foundation for making the compiler more accessible.

tvm/ir
------
The `tvm/ir` folder contains the unified data structure and interfaces across all IR function variants.
The components in `tvm/ir` are shared by `tvm/relax` and `tvm/tirx`, notable ones include

- IRModule
- Type
- PassContext and Pass
- Op

Different variants of functions(e.g. relax.Function and tirx.Function) can co-exist in an IRModule.
While these variants may not have the same content representation, they use the same data structure to represent types.
As a consequence, we use the same data structure to represent function (type) signatures of these variants.
The unified type system allows one function variant to call another function
once we clearly define the calling convention. This opens doors for future cross-function-variant optimizations.

We also provide a unified PassContext for configuring the pass behavior, and common composite passes to execute a pass pipeline.
The following code snippet gives an example of PassContext configuration.

.. code-block:: python

    # configure the behavior of the tirx.UnrollLoop pass
    with tvm.ir.transform.PassContext(config={"tirx.UnrollLoop": { "auto_max_step": 10 }}):
        # code affected by the pass context


Op is the common class to represent all system-defined primitive operator/intrinsics.
Developers can register new Ops as well as their additional attributes(e.g. whether the Op is elementwise) to the system.

.. toctree::
   :maxdepth: 1

   pass_infra

tvm/script (TVMScript)
----------------------

TVMScript is a Python-based DSL for writing TVM IR. It allows users to define ``IRModule``\ s
— containing both Relax functions and TIR ``Function``\ s — using familiar Python syntax with
three import aliases: ``I`` (module-level), ``T`` (TIR), and ``R`` (Relax). Although TVMScript
uses Python syntax, it is not executed by the Python interpreter — decorators like
``@I.ir_module``, ``@T.function``, and ``@R.function`` extract the Python AST and transform
it into TVM IR through a parser and IR builder pipeline.

TVMScript also supports **roundtrip**: any ``IRModule`` can be printed back to TVMScript via
``mod.script()`` and re-parsed to produce a structurally equivalent module. See
:ref:`tvmscript-arch` for the full architecture documentation, including the parser dispatch
mechanism, IR builder frame stack, printer pipeline, and syntax reference.

.. toctree::
   :maxdepth: 1

   tvmscript


tvm/target
----------
The target module contains all the code generators that translate an IRModule to a target tvm_ffi.Module.
It also provides a common `Target` class that describes the target.

Targets can be constructed from a registered tag, a configuration dictionary, or a tag with attribute overrides:

.. code-block:: python

    from tvm.target import Target

    # From a registered tag
    target = Target("nvidia/nvidia-a100")

    # From a config dictionary
    target = Target({"kind": "cuda", "arch": "sm_80"})

    # From a tag with attribute overrides
    target = Target({"tag": "nvidia/nvidia-a100", "l2_cache_size_bytes": 12345})

Use ``Target.list_kinds()`` to see all available target kinds, and ``target.attrs`` to inspect
target attributes.

The compilation pipeline can be customized according to the target by querying the attribute information
in the target and builtin information registered to each target id(cuda, opencl).

.. toctree::
   :maxdepth: 1

   device_target_interactions

tvm/relax
---------

Relax is the high-level IR used to represent the computational graph of a model. Various optimizations are defined in ``relax.transform``.
Note that Relax usually works closely with the TensorIR IRModule, most of the transformations are applied on both Relax and TensorIR functions
in the IRModule. Please refer to the :ref:`Relax Deep Dive <relax-deep-dive>` for more details.

tvm/tirx
--------

``tirx`` contains the core IR definitions and lowering infrastructure
for TensorIR (split from the former ``tir`` module). ``tirx::Function``
represents low-level tensor functions that can be transformed by tirx passes.

The tirx module includes:

- IR data structures (Function, Buffer, SBlock, expressions, statements).
- Analysis passes in ``tirx/analysis``.
- Transformation and lowering passes in ``tirx/transform``.

tvm/s_tir
---------

``s_tir`` (Schedulable TIR, split from the former ``tir`` module) contains
schedule primitives and auto-tuning tools that operate on ``tirx::Function``:

- Schedule primitives to control code generation (tiling, vectorization, thread
  binding) in ``s_tir/schedule``.
- Builtin tensor intrinsics in ``s_tir/tensor_intrin``.
- MetaSchedule for automated performance tuning.
- DLight for pre-defined, high-performance schedules.

Please refer to the :ref:`TensorIR Deep Dive <tensor-ir-deep-dive>` for more details.

tvm/sym
-------

This module is closely tied to TensorIR. One of the key problems in the low-level code generation is the analysis of the indices'
arithmetic properties — the positiveness, variable bound, and the integer set that describes the iterator space. sym module provides
a collection of tools that do (primarily integer) analysis. A TensorIR pass can use these analyses to simplify and optimize the code.

tvm/te and tvm/topi
-------------------

TE stands for Tensor Expression. TE is a domain-specific language (DSL) for describing tensor computations. Importantly, a tensor expression
itself is not a self-contained function that can be stored into IRModule. We can use ``te.create_function`` to convert a tensor expression to a ``tirx::Function``
and then integrate it into the IRModule.

While possible to construct operators directly via TensorIR or tensor expressions (TE) for each use case, it is tedious to do so.
`topi` (Tensor operator inventory) provides a set of pre-defined operators defined by numpy and found in common deep learning workloads.
