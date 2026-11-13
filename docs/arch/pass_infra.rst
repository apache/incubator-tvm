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

.. _pass-infra:

Pass Infrastructure
===================

A pass transforms an ``IRModule`` into another ``IRModule``. ``PassContext``
provides scoped configuration and instrumentation, and ``Sequential`` composes
passes in an explicit order. The same utilities support transformations across
IR dialects and allow pass implementations in C++ or Python.

Passes do not declare dependencies or discover prerequisite passes by name.
A pipeline supplies all of its passes in the order they should execute.

C++ Backend
~~~~~~~~~~~

``PassInfo`` contains a pass's ``name`` and ``opt_level``. The optimization
level controls whether a pass in a ``Sequential`` runs under the current
``PassContext``. See `include/tvm/ir/transform.h`_ for the core interfaces.

.. code:: c++

    class PassInfoObj : public Object {
      int opt_level;
      ffi::String name;
    };

PassContext
^^^^^^^^^^^

``PassContext`` configures transformations with an optimization level, explicit
``required_pass`` and ``disabled_pass`` names, pass-specific configuration, and
instruments. These name lists control passes already present in a pipeline;
they do not create or insert passes. See :ref:`pass_instrument_cpp_backend`
for instrumentation behavior.

This class is designed for users to conveniently write the Python ``with``
syntax to perform optimizations under a certain configuration. In addition, the
users can obtain the context that is available within a certain program scope in
a thread-safe way through ``PassContext::Current()``, since a thread-local store
``PassContextThreadLocalStore`` is used to hold the created pass context
objects. Examples will be provided later to show how we can use both the C++ and
Python APIs to create a compilation pipeline using pass context.

.. code:: c++

    class PassContextObj : public Object {
     public:
      int opt_level{2};
      ffi::Array<ffi::String> required_pass;
      ffi::Array<ffi::String> disabled_pass;
      ffi::Map<ffi::String, Any> config;
      ffi::Array<PassInstrument> instruments;
    };

    class PassContext : public ObjectRef {
     public:
      TVM_DLL static PassContext Create();
      TVM_DLL static PassContext Current();
      TVM_DLL void InstrumentEnterPassContext();
      TVM_DLL void InstrumentExitPassContext();
      TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
      TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;
      /* Other fields are omitted. */

     private:
      // The entry of a pass context scope.
      TVM_DLL void EnterWithScope();
      // The exit of a pass context scope.
      TVM_DLL void ExitWithScope();

      // Classes to get the Python `with` like syntax.
      friend class tvm::With<PassContext>;
    };

    struct PassContextThreadLocalEntry {
      /*! \brief The default pass context. */
      PassContext default_context;
      /*! \brief The current pass context. */
      std::stack<PassContext> context_stack;
      PassContextThreadLocalEntry() {
        default_context = PassContext(ffi::make_object<PassContextObj>());
      }
    };

Pass Constructs
^^^^^^^^^^^^^^^

The pass infra is designed in a hierarchical manner, and it could work at
different granularities of Relax/TensorIR programs. A pure virtual class ``PassObj`` is
introduced to serve as the base of the different optimization passes. This class
contains several virtual methods that must be implemented by the
subclasses at the level of modules, functions, or sequences of passes.

.. code:: c++

    class PassObj : Object {
      virtual PassInfo Info() const = 0;
      virtual Module operator()(const IRModule& mod
                                const PassContext& pass_ctx) const = 0;
    };

The functor shows how a pass must be realized, i.e. it always works on a
:py:class:`IRModule` under a certain context. All passes are designed in a ``Module`` to ``Module``
manner. Therefore, optimizations governed by the pass infra will
always update the whole module.

Several subclasses have been created to implement different types of
optimization passes, e.g., function-level passes, module-level passes, and
sequential passes. Each adapts its transformation to the same module-to-module
interface. Their definitions can be found in `src/ir/transform.cc`_.

Module-Level Passes
^^^^^^^^^^^^^^^^^^^

Module level passes are geared mainly for global and inter-procedural
optimizations (IPO), which are similar to the module pass used in LLVM. Some
typical passes in Relax that need the global picture of a module, such as
A-normal form conversion and lambda lifting, etc., fall into this set. At this
level, users can even add and/or delete functions in a module. Note that all
passes

.. code:: c++

    class ModulePassObj : PassObj {
      PassInfo pass_info;
      std::function<Module(Module, PassContext)> pass_func;
      Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
      // Other members/methods are omitted
    };

``pass_info`` maintains the information needed by a module-level pass.
``pass_func`` sketches the real optimization. For example, we may need to
perform dead code elimination on the module. We could implement the algorithm in
the ``pass_func`` and let it run on a module. It will then remove the dead code
including the unused functions in the module. Note that this field is designed
as a packed function, which enables the implementation of the optimization in
both C++ and Python.

Function-Level Passes
^^^^^^^^^^^^^^^^^^^^^

Function-level passes are used to implement various intra-function level
optimizations for a given Relax/TensorIR module. It fetches one function at a time from
the function list of a module for optimization and yields a rewritten Relax
``Function`` or TensorIR ``Function``. Most of passes can be classified into this category, such as
common subexpression elimination and inference simplification in Relax as well as vectorization
and flattening storage in TensorIR, etc.

Note that the scope of passes at this level is either a Relax function or a TensorIR primitive function.
Therefore, we cannot add or delete a function through these passes as they are not aware of
the global information.

.. code:: c++

    class FunctionPassObj : PassObj {
      PassInfo pass_info;
      std::function<Function(Function, Module, PassContext)> pass_func;
      Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
      bool SkipFunction(const Function& func) const;
      // Other members/methods are omitted...
    };

``pass_info`` is identical to what we just described in the module pass.
``pass_func`` takes a function for optimization, it also needs a module as we
may use it for reporting errors. A function could be annotated with
"SkipOptimization" so that it will be ignored during optimization.

Sequential Passes
^^^^^^^^^^^^^^^^^

``Sequential`` applies its passes in the supplied order. A nested
``Sequential`` preserves the same ordered composition.

.. code:: c++

    class SequentialObj : PassObj {
      PassInfo pass_info;
      // Passes need to be executed.
      ffi::Array<Pass> passes;
      Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
    };

The following code shows how individual passes in a sequential pass are invoked.
Essentially, we sequentially execute each pass in a sequential pass using the
order that they were appended to the pass list.

.. code:: c++

    Module SequentialObj::operator()(const Module& module,
                                      const PassContext& pass_ctx) const {
      Module mod = module;
      for (const Pass& pass : passes) {
        TVM_FFI_ICHECK(pass.defined()) << "Found undefined pass for optimization.";
        const PassInfo& pass_info = pass->Info();
        if (!pass_ctx.PassEnabled(pass_info)) continue;
        mod = pass(mod, pass_ctx);
      }
      return mod;
    }

Upon the invocation of a pass, we first check if this pass is enabled. This is
done by first checking if the pass is explicitly disabled by a user, followed by
inspecting if it is specified as a required pass by the user. If it is still
undetermined whether this pass is enabled, its ``opt_level`` will be checked.
This pass will be enabled and therefore executed only when its optimization
level is at most the configured optimization level in the pass context.

Callers construct the passes directly, including any prerequisites, before
creating a sequence. For example:

.. code:: python

    pipeline = tvm.ir.transform.Sequential([
        relax.transform.Normalize(),
        relax.transform.FoldConstant(),
    ])

Some helper functions are provided to create each type of these aforementioned
passes. These helpers are also exposed to the Python frontend for users to
favorably use Python APIs to create a specific pass object.

.. code:: c++

    Pass CreateFunctionPass(
        std::function<Function(Function, IRModule, PassContext)> pass_func,
        int opt_level,
        ffi::String name);

    Pass CreateFunctionPass(
        std::function<Function(Function, IRModule, PassContext)> pass_func,
        int opt_level,
        ffi::String name);

    Pass CreateModulePass(
        std::function<IRModule(IRModule, PassContext)> pass_func,
        int opt_level,
        ffi::String name);

    Pass Sequential(tvm::ffi::Array<Pass> passes, PassInfo pass_info);

Pass Registration
^^^^^^^^^^^^^^^^^

We've covered the concept of different level of passes and the context used for
compilation. It would be interesting to see how easily users can register
a pass.  Let's take const folding as an example. This pass has already been
implemented to fold constants in a Relax function (found in
`src/relax/transform/fold_constant.cc`_).

An API was provided to perform the ``Expr`` to ``Expr`` transformation.

.. code:: c++

    Expr FoldConstant(const Expr& expr);

In order to register this pass to the pass infra, we first need to decide at
which level this pass will be performed. As const folding happens on individual
functions, we should intuitively create a ``FunctionPass`` for it through
``CreateFunctionPass``. The ``pass_func`` is returned as a packed function that
invokes the ``Expr`` to ``Expr`` API on each function in an ``IRModule``.

Meanwhile, a pass API endpoint is registered with the name
``"relax.transform.FoldConstant"``. This pass, therefore, becomes an entry in the
registry that exposes the factory to Python. C++ callers use the declared
factory directly.

.. code:: c++

    namespace transform {

    Pass FoldConstant() {
      auto pass_func =
          [=](Function f, IRModule m, PassContext pc) { return ConstantFolder::Fold(f, m); };
      return CreateFunctionPass(pass_func, 0, "FoldConstant");
    }

    TVM_FFI_STATIC_INIT_BLOCK() {
      namespace refl = tvm::ffi::reflection;
      refl::GlobalDef().def("relax.transform.FoldConstant", FoldConstant);
    }

    }  // namespace transform

To allow other C++ modules to apply this pass, we declare a free function in
`include/tvm/relax/transform.h`_ as the following:

.. code:: c++

    TVM_DLL Pass FoldConstant();

.. _pass_instrument_cpp_backend:

Pass Instrument
^^^^^^^^^^^^^^^

Pass Instrument is a mechanism to analyze the pass itself. For example,
we can use the infrastructure to know how much time and memory a pass requires
or how a pass can transform the IR module.

We introduce four instrument points in the life-cycle of ``PassContext``.

.. code:: c++

    TVM_DLL void InstrumentEnterPassContext();
    TVM_DLL void InstrumentExitPassContext();
    TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
    TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;

``InstrumentEnterPassContext`` is called immediately when entering the scope
of the ``PassContext`` instance.

``InstrumentExitPassContext`` is called when leaving the scope of ``PassContext``,
or exceptions occur during the execution of passes.
This method is also called when instruments is being overridden by ``override_instruments`` in :py:class:`tvm.ir.transform.PassContext`.
See :ref:`pass_instrument_overriden`.

``InstrumentBeforePass`` is called before execution.
``InstrumentAfterPass`` is called after execution if the pass should be run. The behavior is like:

.. code:: c++

      if (pass_ctx.InstrumentBeforePass(ir_module, pass_info)) {
        new_ir_module = run_pass(ir_module, pass_ctx);
        pass_ctx.InstrumentAfterPass(new_ir_module, pass_info);
        return new_ir_module;
      }

The ``PassInstrument`` interface allow you to run arbitrary code inside above four methods.
Multiple ``PassInstrument`` instances can be registed into a single
``PassContext``. ``PassInstrument`` instances are called sequentially in the order of
``instruments`` argument passed to ``PassContext``.

``PassInstrument`` provides following interfaces:

.. code:: c++

    namespace transform {

    class PassInstrumentObj : public Object {
     public:
      ffi::String name;
      virtual void EnterPassContext() const = 0;
      virtual void ExitPassContext() const = 0;
      virtual bool ShouldRun(const IRModule& mod, const PassInfo& info) const = 0;
      virtual void RunBeforePass(const IRModule& mod, const PassInfo& info) const = 0;
      virtual void RunAfterPass(const IRModule& mod, const PassInfo& info) const = 0;
      /* Other fields are omitted. */
    };

    class PassInstrument : public ObjectRef {
     public:
      TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PassInstrument, ObjectRef, PassInstrumentObj);
    };

    }  // namespace transform

Python frontend are provided to implement ``PassInstrument`` quickly. See :ref:`pass_instrument_py_frontend`.

Within a ``PassContext``, the call sequence of a ``PassInstrument`` instance is like:

::

    with PassContext(instruments=[pi]) # pi = a PassInstrument implementation.
        pi.EnterPassContext()

        if pi.ShouldRun(Pass1):
            pi.RunBeforePass()
            Pass1()
            pi.RunAfterPass()

        if pi.ShouldRun(Pass2):
            pi.RunBeforePass()
            Pass2()
            pi.RunAfterPass()

        pi.ExitPassContext()

Here is a brief introduction of relations between ``PassInstrument`` interfaces
and ``PassContext`` methods. See (`src/ir/transform.cc`_) for more details.

- ``InstrumentEnterPassContext``

  * ``EnterPassContext()`` is executed in the order of ``instruments`` passed to the ``PassContext``.
  * When an exception raises, ``PassContext`` disable the pass instrumentation
    by clearing all registered ``PassInstrument`` instances.
  * Then ``PassContext`` execute ``ExitPassContext()`` method of each ``PassInstrument``
    instances which successfully finished ``EnterPassContext()``
  * For example, if ``PassInstrument`` A, B, and C are registered to a ``PassContext``
    and A finished ``EnterPassContext()`` while B throws an exception, then C
    is never executed; ``ExitPassContext()`` of A is executed.

- ``InstrumentExitPassContext``

  * ``ExitPassContext()`` of each ``PassInstrument`` instances are executed in
    the order of ``instruments`` passed to the ``PassContext``.
  * While an exception occurs, ``instruments`` is cleared.
  * ``PassInstrument`` Instances registered after the one throwing exceptions do not execute ``ExitPassContext``.

- ``InstrumentBeforePass``

  * ``ShouldRun`` is executed if the pass is not listed as a required pass.
  * ``RunBeforePass`` is executed in the order of ``instruments`` if the pass is not blocked by ``ShouldRun``.
  * Note that ``InstrumentBeforePass`` returns a boolean indicating whether or not the pass should be run.
  * When an exception occur, it is thrown immediately.
    We rely on Python Context Manager to exit ``PassContext`` safely
    (meaning ``ExitPassContext`` of each instruments will be run. For C++, please refer to `include/tvm/support/with.h`_.)

- ``InstrumentAfterPass``

  * ``RunAfterPass`` is executed in the order of ``instruments`` passed to the ``PassContext``.
  * When an exception occur, it is thrown immediately.
    We rely on Python Context Manager or ``With`` class(`include/tvm/support/with.h`_) to exit ``PassContext`` safely

Built-in Instrument
^^^^^^^^^^^^^^^^^^^

There are several built-in instruments.

- PassTimingInstrument (see `src/ir/instrument.cc`_)

  * Profile the execution time of passes.

- PrintBeforeAll (see `python/tvm/ir/transform/instrument.py`_)

  * Print the IR module and pass info before each pass executes.

- PrintAfterAll (see `python/tvm/ir/transform/instrument.py`_)

  * Print the IR module and pass info after each pass executes.

- PassPrintingInstrument (see `python/tvm/ir/transform/instrument.py`_)

  * Selectively print the IR module before or after specific named passes.

- DumpIR (see `python/tvm/ir/transform/instrument.py`_)

  * Dump the IR module to files after each pass executes.

Python Frontend
~~~~~~~~~~~~~~~

Only some simple APIs are needed for the frontend side. For example, we can
provide users the following APIs to create and execute a pass (full
implementation is provided in `python/tvm/relax/transform/transform.py`_ and
`python/tvm/ir/transform/core.py`_). The backend
receives the information and decides which function it should use to create
a Pass object.

PassContext
^^^^^^^^^^^

Python frontend provides a wrapper for the ``PassContext`` to enable the
``with`` syntax by overriding ``__enter__`` and ``__exit__``. A ``current``
static method is offered for users to get the context that is in use under
a certain scope.

.. code:: python

    @tvm_ffi.register_object("transform.PassContext")
    class PassContext(tvm_ffi.Object):
        def __enter__(self):
            _transform.EnterPassContext(self)
            return self

        def __exit__(self, ptype, value, trace, config):
            _transform.ExitPassContext(self)

        @staticmethod
        def current():
            """Return the current pass context."""
            return _transform.GetCurrentPassContext()

A ``PassContext`` is used to configure the compilation options, including the
optimization level and required/disabled passes. It can also take a dictionary
of configs so that different passes can conveniently fetch the passed data, such
as fallback device info and loop-unrolling limits. Register each configuration
key and value type directly in a static initialization block:

.. code:: c++

    TVM_FFI_STATIC_INIT_BLOCK() {
      tvm::transform::PassContext::RegisterConfigOption<UnrollLoopConfig>("tirx.UnrollLoop");
    }

``GetConfig<T>(key)`` returns an optional value. Scalar and target-dependent
defaults are explicit at the call site. ``GetConfigOrDefault<TConfig>(key)``
returns a configured object or, only when absent, constructs a fresh object
using its reflection-defined defaults. Both APIs take only the key.

.. code:: c++

    auto ctx = tvm::transform::PassContext::Current();
    bool noalias = ctx->GetConfig<bool>("tirx.noalias").value_or(true);
    auto unroll = ctx->GetConfigOrDefault<UnrollLoopConfig>("tirx.UnrollLoop");

Please refer to `src/tirx/transform/unroll_loop.cc`_ for more details.

.. _pass_instrument_py_frontend:

Pass Instrument
^^^^^^^^^^^^^^^

One can implement a ``PassInstrument`` by using the ``pass_instrument``
decorator (`python/tvm/ir/transform/core.py`_) or by subclassing
``tvm.ir.transform.PassInstrument``. Implement any of the following callbacks:

- ``enter_pass_ctx``

  * This method is run when entering ``PassContext``.

- ``exit_pass_ctx``

  * This method is run when exiting ``PassContext``.

- ``should_run``

  * This method is run before a pass is executed, returning a boolean
    indicating whether or not the pass should be run.

- ``run_before_pass``

  * If a pass should be run, this method is run just before pass execution.

- ``run_after_pass``

  * This method is run right after a pass has been executed.

``PassInstrument`` instances can be registered through ``instruments`` argument in
:py:class:`tvm.ir.transform.PassContext`.

Core interfaces are exported from ``tvm.ir.transform``. Concrete tools, including
``PassTimingInstrument`` and ``DumpIR``, live in ``tvm.ir.transform.instrument``;
see `python/tvm/ir/transform/instrument.py`_ for examples.

.. _pass_instrument_overriden:

Override Instruments in Current PassContext
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``override_instruments`` method is provided to override the ``instruments`` of current ``PassContext``.
For example, if passes are run without explicitly creating a new ``PassContext``,
one can still register ``PassInstrument`` into the global ``PassContext`` by:

.. code:: python

    cur_pass_ctx = tvm.ir.transform.PassContext.current()
    # override PassInstrument instances
    cur_pass_ctx.override_instruments([pass_inst])
    mod = pass_seq(mod)
    result = pass_inst.get_result()

Note that when ``override_instruments`` is called, the ``exit_pass_ctx`` method of
old ``PassInstrument`` instances are called. Then the ``enter_pass_ctx`` method of
new ``PassInstrument`` are called.

.. _include/tvm/ir/transform.h: https://github.com/apache/tvm/blob/main/include/tvm/ir/transform.h

.. _include/tvm/support/with.h: https://github.com/apache/tvm/blob/main/include/tvm/support/with.h

.. _src/relax/ir/transform.cc: https://github.com/apache/tvm/blob/main/src/relax/ir/transform.cc

.. _src/ir/transform.cc: https://github.com/apache/tvm/blob/main/src/ir/transform.cc

.. _src/ir/instrument.cc: https://github.com/apache/tvm/blob/main/src/ir/instrument.cc

.. _src/relax/transform/fold_constant.cc: https://github.com/apache/tvm/blob/main/src/relax/transform/fold_constant.cc

.. _python/tvm/relax/transform/transform.py: https://github.com/apache/tvm/blob/main/python/tvm/relax/transform/transform.py

.. _include/tvm/relax/transform.h: https://github.com/apache/tvm/blob/main/include/tvm/relax/transform.h

.. _python/tvm/ir/transform/core.py: https://github.com/apache/tvm/blob/main/python/tvm/ir/transform/core.py

.. _python/tvm/ir/transform/instrument.py: https://github.com/apache/tvm/blob/main/python/tvm/ir/transform/instrument.py

.. _src/tirx/transform/unroll_loop.cc: https://github.com/apache/tvm-tirx/blob/main/src/tirx/transform/unroll_loop.cc

.. _use pass infra: https://github.com/apache/tvm/blob/main/docs/how_to/tutorials/customize_opt.py
