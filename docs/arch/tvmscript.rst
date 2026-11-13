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

.. _tvmscript-arch:

TVMScript
=========

TVMScript expresses TVM IR with Python syntax. A shared frontend translates source
into a Python builder program; executing that program constructs IR. A separate
printer converts IR back into readable, parseable source.

.. code-block:: text

   Source + definition context
       → Python AST → syntax transpiler → Python builder program → TVM IR
   TVM IR
       → DocTranslator → Doc tree + origins → diagnostic paths → Python text

Source and frontend
-------------------

``tvm.script.parser`` accepts Python functions, classes and source strings through
public decorators, ``tvm.script.parse`` and ``tvm.script.from_source``. The frontend
acquires source and locations, captures the definition's globals and closure bindings,
and composes the generated callable with that environment. String callers provide
external bindings through ``extra_vars``.

Standalone function decorators construct IR immediately. Within an IR-module class,
function decorators retain definitions until module construction, which can declare
signatures before building bodies. Captured Python values belong to the source
context; symbolic IR values are created and resolved by builders.

Function construction, JIT and macro decorators require ``@`` application at the
function definition site. Later application to an existing function is rejected.
Use a registered namespace, such as ``@T.function`` or ``@Ts.function(private=True)``.
Python definitions support namespace aliases such as ``Alias = T``; bare callable
aliases and preconfigured decorator aliases are unsupported. Source strings resolve
namespace aliases from imports or ``extra_vars``, not executable prefix assignments.

``GeneratedBuilder`` records connect generated body helpers to original functions and
identify bindings preserved during recomposition. Original-name wrapper parameters
capture definition values as defaults. Execution globals remain separate from the
locals that evaluate those defaults. Declaration and annotation helpers bind snapshot
values under their original names, so Python handles parameter and local shadowing.
Missing snapshot values raise only when their annotation expressions read them.
Class setup and declaration snapshots execute in source order before function bodies.
Body globals, closures and local shadowing retain their Python scope. Missing values
are read only when needed, including conditional and optional annotations.

Annotations read concrete values from their definition scope, preserving missing-name
errors when a value is used. Create external symbols with ``n = I.dynamic("n")``
or the identical ``T.dynamic`` and ``Ts.dynamic`` constructors. Each call creates a
fresh native variable, defaulting to int32. Reuse that object in ordinary Python shape,
stride and offset expressions to share identity; strings in those fields are not
parsed as expressions. On Python 3.12+, explicit headers such as
``def f[n, k: T.int64](...)`` declare local symbols; ``n: int`` retains the int32
default. Relax symbolic shapes require explicit int64 symbols, such as
``I.dynamic("n", "int64")`` or ``def f[n: T.int64](...)``; Relax integer shape
constants continue to normalize to int64. Use ``from __future__ import annotations``
to defer eager Python evaluation of header symbols. Explicit whole quoted annotations are rejected in script source;
normal string arguments inside annotation constructors remain valid. Captured runtime ``typing.TypeVar``
objects are not script symbols; ordinary Python typing uses remain unaffected.

An explicit scalar annotation ``n: n`` preserves a captured native symbol's identity.
An independently typed parameter such as ``n: T.int32`` and ordinary body locals
shadow definition captures normally. Annotation classes, Python unions and deferred
return-constructor evaluation retain their builder behavior.

Syntax and construction protocol
--------------------------------

The syntax transpiler rewrites a fresh Python AST using scope and declaration facts
from a prescan. A registered decorator selects the construction namespace. Assignments
call binding hooks, expression statements call emission hooks, and control flow opens
builder frames. The namespace owns the meaning of these operations and the supported
IR constructs. Standalone decorators pass already-evaluated ``root_function_kwargs``
so option expressions execute once. Nested functions and module members use their own
decorator options. Builder hooks own option defaults; ``check_well_formed`` remains a
separate parser setting for standalone functions. Set it to ``False`` on a function
decorator to skip that function's validation. ``@I.ir_module`` constructs a module
without automatic whole-module validation. Validate completed modules explicitly
with the language's ``analysis.check_well_formed(module)`` API. These checks raise on
invalid IR by default; TIRx also supports ``assert_mode=False`` for a boolean result.

``tvm.script.parser.protocol_registry`` records syntax policies under registered
canonical namespace paths rooted at ``tirx``, ``relax``, ``ir`` and ``s_tir``.
Source code selects its own aliases through imports or explicit environments; the parser
does not provide implicit ``T``, ``R``, ``I`` or ``Ts`` bindings. For example,
``from tvm.script import tirx as X`` makes ``X.function`` resolve to
``tirx.function`` without changing the callable. Printed scripts show suggested imports
as comments: include those imports in executable source or pass equivalent
``extra_vars`` when parsing a printed script. The public registration helpers
identify scalar annotations and mutable
declarations. Entry factories populate ``DEFINITION_KIND`` with ``DefinitionKind``
values: ``FUNCTION`` for regular and JIT IR definitions, ``MACRO`` for inline and macro
expansion, and ``PYTHON`` for retained ``I.pyfunc`` runtime callables. Macro bodies
construct IR in the caller's context; Python bodies retain ordinary execution and do
not need builder loc context.
All calls retain their source context, with binding hooks owning result attachment.
Symbolic shapes use concrete expressions. Source aliases resolve to those paths;
ordinary Python calls remain calls in the generated program. Explicit ``constexpr``
markers from the shared builder select host control flow during construction.

JIT supplies ``const_args``, a mapping from parameter names to fixed values. Explicit
``None`` values mark absent optional arguments and skip their annotation evaluation.
Generated code reads the mapping through a fresh ``_const_args`` name. An empty map
still selects root JIT construction; an absent map selects ordinary parsing. Syntax
translation does not inspect these values to select a branch.

Generated operations retain source locations. Syntax restrictions raise source-located
``SyntaxError`` exceptions; builder and Python helper errors retain their original
exception types. Temporary parse state is released when construction finishes or fails.

Frame-based construction
------------------------

The IR builder maintains an active stack of frames, such as module, function and loop
frames. Entering a frame establishes its scope; exiting finalizes its IR and attaches it
to its parent. Frames own parameters, symbols and construction state. Builder hooks own
value binding, type rules and validation of completed IR. The shared
``resolve_global_info_args`` decorator resolves named metadata arguments after
ordinary Python argument evaluation; dialect callbacks own selector syntax and
lookup in the active module.

Public script namespaces expose decorators and construction operations. The same
underlying builders can also be used directly from Python. The transpiler therefore
needs no separate mutable IR representation: it generates calls to this construction
protocol. ``script.namespace`` records the construction namespace for printing
and dialect lowering. S-TIR functions use ``"s_tir"``; their builders own block
completion and validation.

Ordinary IR constructors can be used directly in parsed source. Shared exports
such as ``I.Call`` and ``T.Range`` use the same constructor contracts as
``tvm.ir.Call`` and ``tvm.ir.Range``, including keyword arguments, source locations
and validation. ``Call(op, args, ..., ty=ty)`` preserves an explicit result type, including
``Type.missing()``. Omitting ``ty`` or passing ``ty=None`` uses available inference and retains
``Type.missing()`` when no deduction is available; inference errors propagate.
Construction permits provisional IR, and ``call.validate()`` checks the registered
operator contract separately. Raw printed calls use ``I.Call`` with an explicit
stored result type to preserve every field. Canonical named operations use the same
``args``, ``attrs``, ``ty_args`` and ``ty`` construction contract. Concrete printer
hooks handle representation-changing syntax; ordinary calls share one formatter.

Operations likewise retain their normal argument contracts. A dtype inferred from
operands is not an extra ``dtype`` keyword; operations with an explicit dtype
parameter accept it normally. Annotation shorthand, module metadata selectors,
frame construction and variadic dtype positioning remain explicit builder adapters.
They do not change the underlying IR constructor's validation or consult builder
state from ordinary construction.

Printing and round trips
------------------------

``DocTranslator`` invokes native type and operation hooks to produce a ``Doc`` tree
of expressions and statements. The translation engine tracks scopes, names and each
Doc's original IR object. The script entry point maps these origins to diagnostic
paths; the private Doc printer formats the tree, annotations and underlines as Python
text. Printer configuration stays read-only throughout. The Python document printer
helper also formats an existing Doc directly for document-level tests. This tree is
separate from the parser's Python AST.

The public ``tvm::Script`` text entry point is declared in
``tvm/script/printer/printer.h``. Its orchestration and diagnostic path mapping
live in ``src/script/printer/printer.cc``; ``doc_translator.h`` exposes the
IR-to-Doc translation protocol.

For example, a small function can be authored, printed and parsed again:

.. code-block:: python

   import tvm
   from tvm.script import tirx as T

   @T.function
   def increment(A: T.Tensor((4,), "float32")):
       for i in T.serial(4):
           A[i] = A[i] + 1.0

   text = increment.script()
   reparsed = tvm.script.from_source(text, extra_vars={"T": T})
   tvm.ir.assert_structural_equal(increment, reparsed)

Printed source uses canonical forms rather than preserving the original spelling.
Round trips require printable IR and any external objects needed by the source;
the text does not serialize arbitrary Python state.

Namespace extension
-------------------

A language variant supplies construction hooks and a public script namespace, then
registers its aliases with ``register_namespace``. ``register_namespace_initializer``
supports lazy setup. ``tvm.script.register_dialect`` exposes a package through the
public script namespace. Syntax policies are registered beside the operations that
need them. IR printing uses ``FDocTranslate`` hooks from
``tvm/script/printer/doc_translator.h``, registered through the existing type or
operation attributes. A hook returns an expression Doc or emits completed statements
through its translator context.

This division keeps source acquisition, syntax translation and formatting shared.
Language-specific construction and validation remain with the namespace and its
builders.
