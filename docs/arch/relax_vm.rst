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

.. _relax-vm-arch:

Relax Virtual Machine
=====================

This document explains the Relax VM architecture in detail, covering the compilation pipeline
from Relax IR to bytecode, the instruction set, the execution model, and the Python-level user
interface.

Overview
--------

The end-to-end flow from model to execution is:

1. **Relax IR** — a high-level computational graph (``relax.Function`` inside an ``IRModule``).
2. **Compilation** — ``tvm.compile()`` applies the Relax transformation pipeline, then invokes
   ``VMCodeGen`` to translate each Relax function into bytecode instructions.
3. **Linking** — TIR functions are compiled to native kernels (via LLVM, CUDA, etc.); the bytecode,
   constant pool, and compiled kernels are packaged together into a ``VMExecutable``.
4. **Execution** — at runtime, a ``VirtualMachine`` loads the executable, initializes devices and
   memory allocators, and runs the bytecode.

.. code-block:: text

   IRModule (Relax + TIR)
        │
        ▼  relax_pipeline (FuseOps, LegalizeOps, ...)
   IRModule (optimized)
        │
        ▼  VMCodeGen
   ExecBuilder (bytecode) + IRModule (TIR only)
        │                        │
        │                        ▼  tirx.compile()
        │                   tvm_ffi.Module (native kernels)
        │                        │
        ▼  VMLink               ▼
   VMExecutable ◄───────── linked together
        │
        ▼  VirtualMachine(exec, device)
   Runtime execution


Compilation: From Relax IR to Bytecode
--------------------------------------

Build entry point
~~~~~~~~~~~~~~~~~

The main entry point is ``tvm.compile()`` (which delegates to ``relax.build()`` in
``python/tvm/relax/vm_build.py``):

.. code-block:: python

   import tvm
   from tvm import relax

   @tvm.script.ir_module
   class MyModule:
       @R.function
       def main(x: R.Tensor((3, 4), "float32")):
           return R.add(x, x)

   target = tvm.target.Target("llvm")
   ex = tvm.compile(MyModule, target)

Internally, ``relax.build()`` performs these steps:

1. Apply the **Relax pipeline** (``relax.get_pipeline("default")``), which includes operator
   legalization, fusion, buffer planning, and other graph-level passes.
2. Create an ``ExecBuilder`` and run **VMCodeGen** (``src/relax/backend/vm/codegen_vm.cc``),
   which walks each ``relax.Function`` and emits bytecode instructions. The Relax functions are
   removed from the IRModule; only TIR functions remain.
3. Compile the remaining TIR functions to native code via ``tirx.compile()``.
4. **Link** the bytecode executable with the compiled native module using ``VMLink``, producing
   a ``VMExecutable``.

Relax functions execute through the VM bytecode dispatch loop.

Bytecode generation
~~~~~~~~~~~~~~~~~~~

The ``CodeGenVM`` class (``src/relax/backend/vm/codegen_vm.cc``) is an ``ExprFunctor`` that visits
each Relax expression and emits instructions through the ``ExecBuilder``:

- Each ``relax.Var`` is mapped to a register.
- Function parameters occupy registers 0 through N-1.
- Each binding in a ``SeqExpr`` generates one or more instructions; the result is stored in a
  new register.
- Function calls (``R.call_tir``, ``R.call_packed``, operator calls) become ``Call`` instructions.
- Conditional expressions (``relax.IfExpr``, written as Python ``if`` in TVMScript) become an ``If``
  instruction followed by ``Goto`` to skip branches.
- The function body ends with a ``Ret`` instruction.


Instruction Set
---------------

The VM uses a **register-based** architecture with an intentionally minimal instruction set.
There are only four opcodes:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Opcode
     - Fields
     - Semantics
   * - ``Call``
     - ``dst``, ``func_idx``, ``num_args``, ``args[]``
     - Call function ``func_idx`` with the given arguments; store the result in register ``dst``.
   * - ``Ret``
     - ``result``
     - Return the value in register ``result`` to the caller.
   * - ``Goto``
     - ``pc_offset``
     - Jump forward or backward by ``pc_offset`` instructions.
   * - ``If``
     - ``cond``, ``false_offset``
     - If register ``cond`` is nonzero, fall through (pc++); otherwise jump by ``false_offset``.

The VM itself performs **no mathematical computation**. All actual work — matrix multiplications,
convolutions, elementwise operations — is carried out by compiled TIR kernels or external
libraries (cuBLAS, cuDNN, etc.), dispatched through ``Call`` instructions.

Instruction encoding
~~~~~~~~~~~~~~~~~~~~

Each instruction argument (``Instruction::Arg``) is a 64-bit word encoded as:

- **Bits [63:56]** — ``ArgKind`` (8 bits): ``kRegister`` (0), ``kImmediate`` (1), ``kConstIdx`` (2),
  or ``kFuncIdx`` (3).
- **Bits [55:0]** — value (56 bits, sign-extended).

Two special register values exist:

- ``kVoidRegister``: indicates "no destination" (the return value is discarded).
- ``kVMRegister``: refers to the VM context pointer itself, passed as the first argument to
  closures.

The instruction stream is stored as a flat ``vector<ExecWord>`` (``instr_data``) with an offset
table (``instr_offset``) for random access.


Executable
----------

A ``VMExecutable`` (``include/tvm/runtime/vm/executable.h``) bundles everything needed for
execution:

- **Function table** (``func_table``): a ``vector<VMFuncInfo>`` describing every function. Each
  entry records the function's kind, name, instruction range (``start_instr`` to ``end_instr``),
  number of arguments, register file size, and parameter names.
- **Constant pool** (``constants``): model weights, shape tuples, and other compile-time constants.
- **Bytecode** (``instr_data`` + ``instr_offset``): the instruction stream.
- **Imported modules**: the compiled TIR kernels and external libraries.

Function kinds
~~~~~~~~~~~~~~

The VM recognizes two function kinds (``VMFuncInfo::FuncKind``):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Kind
     - Description
   * - ``kPackedFunc``
     - An external C/C++ function looked up from imported modules or the global PackedFunc
       registry. Examples: ``vm.builtin.alloc_shape_heap``, ``vm.builtin.match_shape``.
   * - ``kVMFunc``
     - A bytecode-interpreted Relax function. The VM interprets its instructions in ``RunLoop()``.

Serialization
~~~~~~~~~~~~~

The executable supports binary serialization for deployment:

.. code-block:: python

   # Save
   ex.export_library("model.so")

   # Load
   loaded = tvm.runtime.load_module("model.so")
   vm = relax.VirtualMachine(loaded, tvm.cuda())

The binary format includes a magic number (``0xD225DE2F4214151E``), a version string
(currently ``"0.14"``), followed by four sections: globals (the function table), memory scopes,
constant pool, and bytecode. ``AsText()`` and ``AsPython()`` provide human-readable representations
for debugging.


Runtime Execution
-----------------

VM initialization
~~~~~~~~~~~~~~~~~

At runtime, a ``VirtualMachine`` is created and initialized:

.. code-block:: python

   from tvm.relax import VirtualMachine

   vm = VirtualMachine(exec_module, tvm.cuda())

Under the hood:

1. **LoadExecutable**: the bytecode and metadata are loaded from the ``VMExecutable``.
2. **Init**: devices and owning tensor pools are set up. Each device gets a
   ``TensorMemoryPool`` from the shared device factory. A CPU device is always
   added for shape computations.
3. **InitFuncPool**: the function pool is populated — ``kPackedFunc`` entries are resolved from
   imports or the global registry; ``kVMFunc`` entries are wrapped in
   ``VMClosure`` objects.
4. **Constant pool**: model constants are loaded and optionally transferred to the target device.

The bytecode dispatch loop
~~~~~~~~~~~~~~~~~~~~~~~~~~

When a ``kVMFunc`` is invoked, the VM enters ``InvokeBytecode()``:

1. A new ``VMFrame`` is pushed onto the call stack. Each frame contains:

   - A **register file** (``vector<ffi::Any>``) — type-erased slots that can hold tensors,
     shapes, closures, or any TVM object. The size is determined at compile time
     (``VMFuncInfo::register_file_size``).
   - The **return program counter** — where to resume after the function returns.
   - The **caller's return register** — which register in the parent frame receives the result.

2. Function arguments are written to registers 0..N-1.
3. The program counter (``pc_``) is set to the function's ``start_instr``.
4. ``RunLoop()`` executes instructions until a ``Ret`` is encountered:

   - **Call**: resolve arguments (from registers, immediates, constant pool, or function pool),
     invoke the target function via ``InvokeClosurePacked()``, store the result in ``dst``.
   - **Ret**: read the return value from the specified register, write the result to the
     caller's return register, and return from ``RunLoop()`` (the frame is popped by an RAII
     guard when ``InvokeBytecode()`` exits).
   - **Goto**: adjust ``pc_`` by the offset.
   - **If**: check the condition register; if nonzero, fall through; otherwise jump by
     ``false_offset``.

The dispatch loop is implemented in ``src/runtime/vm/vm.cc`` (``VirtualMachineImpl::RunLoop``).

.. code-block:: text

   Frame Stack              Register File (per frame)
   ┌─────────────┐          ┌────┬────┬────┬─────┬────┐
   │  Frame 2    │ ───────► │ R0 │ R1 │ R2 │ ... │ Rn │
   ├─────────────┤          └────┴────┴────┴─────┴────┘
   │  Frame 1    │ ───────► [register file]
   ├─────────────┤
   │  Frame 0    │ ───────► [register file]
   └─────────────┘

VMClosure and function dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions in the VM are stored in a ``func_pool_`` indexed by function table position.
``kVMFunc`` entries are wrapped as ``VMClosure`` objects, while ``kPackedFunc``
entries are stored as plain ``ffi::Function``. A ``VMClosure`` stores:

- ``func_name``: the function's string name.
- ``impl``: a ``ffi::Function`` that takes the VM context pointer as its first argument, followed
  by the actual parameters.

When the VM encounters a ``Call`` instruction, it looks up the function in ``func_pool_`` by
index and dispatches via ``InvokeClosurePacked()``. If the target is a ``VMClosure``, the VM
pointer is prepended to the arguments and ``impl`` is invoked. If it is a plain
``ffi::Function``, it is called directly.

``VMClosure::BindLastArgs`` enables partial application — it creates a new function with
some arguments pre-bound at the end, useful for implementing captured closures in Relax.

Built-in operations
~~~~~~~~~~~~~~~~~~~

The VM relies on several built-in PackedFuncs (registered in ``src/runtime/vm/builtin.cc``)
for runtime support:

- ``vm.builtin.alloc_shape_heap``: allocate workspace for symbolic shape computations.
- ``vm.builtin.match_shape``: validate tensor shapes against expected patterns at runtime,
  supporting assertions (``kAssertEqualToImm``, ``kAssertEqualToLoad``), storing symbolic
  dimensions to the shape heap (``kStoreToHeap``), or no-ops (``kNoOp``).
- ``vm.builtin.make_shape``: construct shape tuples from immediates or heap-loaded values.
- ``vm.builtin.match_prim_value``: validate primitive values (e.g., integers) against expected
  patterns.
- ``vm.builtin.copy``: copy a value into a register. Used in several codegen scenarios:
  materializing non-register arguments (immediates, constants) into registers, ensuring each
  variable binding gets its own register, and merging results from if/else branches.


Python Interface
----------------

Users interact with the VM through ``tvm.relax.VirtualMachine``:

.. code-block:: python

   import tvm
   from tvm import relax
   import numpy as np

   # Compile
   ex = tvm.compile(MyModule, target="llvm")

   # Create VM
   vm = relax.VirtualMachine(ex, tvm.cpu())

   # Direct invocation
   inp = tvm.runtime.tensor(np.random.rand(3, 4).astype("float32"))
   result = vm["main"](inp)

   # Stateful interface (useful for RPC)
   vm.set_input("main", inp)
   vm.invoke_stateful("main")
   output = vm.get_outputs("main")

Key methods:

- ``vm["func_name"](*args)`` — direct invocation, returns the result.
- ``vm.set_input()`` / ``vm.invoke_stateful()`` / ``vm.get_outputs()`` — stateful interface
  that avoids sending output over the wire, useful for RPC-based remote execution.
- ``vm.save_function(func_name, saved_name, *args)`` — pre-bind arguments for repeated calls,
  reducing dictionary lookup overhead during benchmarking.
- ``vm.time_evaluator(func_name, dev)`` — returns a timing function following the same convention
  as ``tvm.runtime.time_evaluator``.
- ``vm.set_instrument(func)`` — register an instrumentation callback that is invoked before/after
  every ``Call`` instruction. The callback can return ``VMInstrumentReturnKind.SKIP_RUN`` to
  skip the call.

Instrumentation
~~~~~~~~~~~~~~~

The VM supports observability via instrumentation:

**Instrumentation** via ``set_instrument()``:

.. code-block:: python

   def my_instrument(func, func_symbol, before_run, ret_value, *args):
       if before_run:
           print(f"About to call: {func_symbol}")
       return VMInstrumentReturnKind.NO_OP

   vm.set_instrument(my_instrument)
   vm["main"](inp)

The instrument function is called before and after every ``Call`` instruction, receiving the
function object, its symbol name, a flag indicating before/after, the return value (only valid
after), and all arguments.


Inspecting Bytecode
-------------------

The executable provides text and Python representations of the compiled bytecode:

.. code-block:: python

   ex = tvm.compile(MyModule, target="llvm")
   print(ex.as_text())    # Human-readable instruction listing
   print(ex.as_python())  # Equivalent Python program
   print(ex.stats())      # Summary statistics

These are invaluable for debugging compilation issues — they show exactly which functions
are called, in what order, and how registers are used.


Source Code Map
---------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Path
     - Contents
   * - ``include/tvm/runtime/vm/bytecode.h``
     - Instruction, Opcode, and Arg definitions
   * - ``include/tvm/runtime/vm/executable.h``
     - VMExecutable, VMFuncInfo, serialization
   * - ``include/tvm/runtime/vm/vm.h``
     - VirtualMachine base class, VMClosure
   * - ``src/runtime/vm/vm.cc``
     - VirtualMachineImpl, RunLoop, InvokeBytecode
   * - ``src/runtime/vm/executable.cc``
     - Serialization/deserialization, text output
   * - ``src/runtime/vm/builtin.cc``
     - Built-in operations (shape matching, allocation)
   * - ``src/relax/backend/vm/codegen_vm.cc``
     - CodeGenVM: Relax IR → bytecode
   * - ``python/tvm/runtime/vm/vm.py``
     - Python VirtualMachine wrapper
   * - ``python/tvm/runtime/vm/param.py``
     - Parameter dictionary serialization
   * - ``python/tvm/relax/vm_build.py``
     - ``relax.build()`` and VMExecutable Python class
