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

.. _relax-dpl:

Dataflow Pattern Language (DPL)
===============================
The Dataflow Pattern Language (DPL) is Relax's built-in facility for
**pattern matching and rewriting** on computation graphs. It lets you describe a
sub-graph structure you are looking for, search for it inside a Relax function,
and optionally replace it with a new structure -- all without hand-writing a
full IR visitor.

DPL is used throughout the TVM stack:

- **Operator fusion** -- ``FuseOpsByPattern`` groups matched operators into a
  single fused function.
- **Backend dispatch** -- CUTLASS, cuBLAS, cuDNN and other backends register
  patterns so the compiler can route sub-graphs to optimized library kernels.
- **Custom graph transforms** -- users write their own patterns and rewriters
  to perform project-specific optimizations.

The typical workflow has three steps:

1. **Build a pattern** that describes the sub-graph shape (e.g. ``matmul`` followed
   by ``add``).
2. **Match** the pattern against Relax IR to locate all occurrences.
3. **Rewrite** each match into a replacement expression.

The public API lives in ``tvm.relax.dpl`` (source: ``python/tvm/relax/dpl/``).


Building Patterns
-----------------
A *pattern* is a lightweight description of what an expression should look like.
Patterns are built by combining small building blocks.

Basic Patterns
~~~~~~~~~~~~~~
The most common leaf patterns are:

- ``wildcard()`` -- matches any expression.
- ``is_op("relax.add")`` -- matches a specific Relax operator.
- ``is_const()`` -- matches any constant value.
- ``is_var(name)`` -- matches a ``Var`` node (optionally with a given name).
- ``is_dfv(name)`` -- matches a ``DataflowVar`` node.
- ``is_gv(name)`` -- matches a ``GlobalVar``.

.. code:: python

    from tvm.relax.dpl import wildcard, is_op, is_const

    # Match any relax.add call, regardless of arguments
    add_pattern = is_op("relax.add")(wildcard(), wildcard())

Call Patterns
~~~~~~~~~~~~~
Calling a pattern as a function produces a ``CallPattern``.  The callee is the
pattern itself, and the positional arguments are patterns for each operand:

.. code:: python

    x = wildcard()
    w = wildcard()

    # Match: relax.matmul(x, w)
    matmul = is_op("relax.matmul")(x, w)

For operators with variadic arguments, pass ``varg_default_wildcard=True`` so
that extra arguments are matched by implicit wildcards:

.. code:: python

    # Match relax.concat with any number of inputs
    concat = is_op("relax.concat")(wildcard(), varg_default_wildcard=True)

DPL also provides specialized helpers for common call patterns:

- ``is_call_tir(func_name, args)`` -- matches ``R.call_tir(func_name, (args...,))``.
- ``is_call_dps_packed(func_name, args)`` -- matches ``R.call_dps_packed``.
- ``is_call_packed(func_name, args)`` -- matches ``R.call_packed``.

.. code:: python

    from tvm.relax.dpl import is_call_tir, wildcard

    # Match a call_tir that calls the function "decode"
    decode = is_call_tir("decode", args=[wildcard(), wildcard()])

Tuple Patterns
~~~~~~~~~~~~~~
``TuplePattern`` matches a Relax tuple with a fixed number of fields.
It supports indexing with ``[]`` to create ``TupleGetItemPattern``:

.. code:: python

    from tvm.relax.dpl import is_tuple, wildcard

    a, b = wildcard(), wildcard()
    tup = is_tuple([a, b])

    # Match: getting the first element from the tuple
    first = tup[0]

Constraints
~~~~~~~~~~~
Any pattern can be further narrowed by attaching constraints:

- ``.has_dtype(dtype)`` -- the matched expression must have the given data type.
- ``.has_shape(shape)`` -- the matched expression must have the given shape.
- ``.has_attr(attrs)`` -- the matched call must carry the given attributes.
- ``.has_ty(ty)`` -- the matched expression must have the given type.

.. code:: python

    # Match a float16 matmul
    fp16_matmul = is_op("relax.matmul")(wildcard(), wildcard()).has_dtype("float16")

Logical Combinators
~~~~~~~~~~~~~~~~~~~
Patterns can be combined with logical operators:

- ``pat_a | pat_b`` -- match if **either** pattern matches (``OrPattern``).
- ``pat_a & pat_b`` -- match if **both** patterns match (``AndPattern``).
- ``~pat`` -- match anything **except** ``pat`` (``NotPattern``).

.. code:: python

    # Match either relu or gelu activation
    activation = is_op("relax.nn.relu")(wildcard()) | is_op("relax.nn.gelu")(wildcard())

Sequence Patterns
~~~~~~~~~~~~~~~~~
When a pattern spans multiple bindings inside a ``DataflowBlock``, use
*sequence operators* to express producer-consumer relationships:

- ``a ^ b`` (``used_by``) -- ``a`` is used by ``b`` (``a`` may also be used
  elsewhere).
- ``a >> b`` (``only_used_by``) -- ``a`` is **only** used by ``b`` (no other
  consumers).

These return a ``PatternSeq`` that can be chained:

.. code:: python

    x = wildcard()
    matmul = is_op("relax.matmul")(x, wildcard())
    add = is_op("relax.add")(matmul, wildcard())

    # matmul result is exclusively consumed by the add
    seq = matmul >> add

High-level Helpers
~~~~~~~~~~~~~~~~~~
``make_fused_bias_activation_pattern`` builds a common
``op -> optional bias -> optional activation`` chain in one call:

.. code:: python

    from tvm.relax.dpl import make_fused_bias_activation_pattern

    # conv2d + bias + relu
    pattern = make_fused_bias_activation_pattern(
        "relax.nn.conv2d",
        with_bias=True,
        activation="relax.nn.relu",
    )


Matching Without Rewriting
--------------------------
Sometimes you only need to **detect** a structure without replacing it.
Every ``DFPattern`` exposes two matching methods:

- ``pattern.match(expr)`` -- returns ``True`` if the pattern matches.
- ``pattern.extract_matched_expr(expr)`` -- returns a
  ``dict[DFPattern, Expr]`` mapping each sub-pattern to the concrete
  expression it matched, or ``None`` on failure.

.. code:: python

    from tvm.relax.dpl import wildcard, is_op

    x = wildcard()
    y = wildcard()
    add_pat = is_op("relax.add")(x, y)

    # Assume `expr` is a Relax expression: R.add(a, b)
    if add_pat.match(expr):
        matched = add_pat.extract_matched_expr(expr)
        # matched[x] -> the expression that matched `x`
        # matched[y] -> the expression that matched `y`

When matching across variable bindings (e.g., ``lv0 = ...; lv1 = f(lv0)``),
the matcher needs a ``var2val`` map so it can see through binding
boundaries.  Use ``tvm.relax.analysis.get_var2val(func)`` to build one:

.. code:: python

    from tvm.relax.analysis import get_var2val

    var2val = get_var2val(func)
    matched = pattern.extract_matched_expr(expr, var2val=var2val)


Rewriting Matched Patterns
--------------------------

``rewrite_call``
~~~~~~~~~~~~~~~~
``rewrite_call`` is the simplest rewrite API.  It walks every expression in a
function, and when the pattern matches, it calls your callback to produce a
replacement.

.. code:: python

    rewrite_call(pattern, rewriter, func) -> Function

The callback signature is:

.. code:: python

    def rewriter(
        matched_expr: Expr,
        matchings: dict[DFPattern, Expr],
    ) -> Expr:
        ...

**Example -- replace** ``reshape(reshape(x, s1), s2)`` **with**
``reshape(x, s2)``:

.. code:: python

    from tvm import relax
    from tvm.relax.dpl import wildcard, is_op, rewrite_call

    inp = wildcard()
    shape1, shape2 = wildcard(), wildcard()
    inner = is_op("relax.reshape")(inp, shape1)
    outer = is_op("relax.reshape")(inner, shape2)

    def rewriter(expr, matchings):
        # Keep the original input but use the outermost target shape
        return relax.op.reshape(matchings[inp], matchings[outer].args[1])

    new_func = rewrite_call(outer, rewriter, func)

``rewrite_call`` is best for **local, single-expression** rewrites.


``rewrite_bindings`` with ``PatternContext``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When a rewrite involves **multiple bindings** across a ``DataflowBlock``
(e.g., merging three separate matmuls into one), use ``rewrite_bindings``
together with ``PatternContext``.

``PatternContext`` enables topological (graph-level) matching on an entire
dataflow block rather than on individual expressions.

.. code:: python

    rewrite_bindings(ctx, rewriter, func) -> Function

The callback receives *variables* rather than expressions:

.. code:: python

    def rewriter(
        matchings: dict[DFPattern, Var],
        bindings: dict[Var, Expr],
    ) -> dict[Var, Expr]:
        ...

- ``matchings[pat]`` returns the **bound variable** (``Var``) whose right-hand
  side matched ``pat``.  The ``Var`` itself carries ``ty`` and can be
  used directly in new expressions.
- ``bindings`` maps each ``Var`` to its bound ``Expr`` (the right-hand side),
  useful when you need to inspect the original expression.

**Example -- merge three parallel matmuls into one**:

.. code:: python

    from tvm.relax import script as R
    from tvm.relax.dpl import wildcard, is_op, rewrite_bindings, PatternContext

    with PatternContext() as ctx:
        inp_pat = wildcard()
        w1, w2, w3 = wildcard(), wildcard(), wildcard()

        matmul1 = is_op("relax.matmul")(inp_pat, w1)
        matmul2 = is_op("relax.matmul")(inp_pat, w2)
        matmul3 = is_op("relax.matmul")(inp_pat, w3)

        def rewriter(matchings, _bindings):
            inp = matchings[inp_pat]
            W1 = matchings[w1]
            W2 = matchings[w2]
            W3 = matchings[w3]
            width = W1.ty.shape[1]

            concat_w = R.concat([W1, W2, W3], axis=1)
            merged = R.matmul(inp, concat_w)

            return {
                matchings[matmul1]: R.strided_slice(
                    merged, axes=[2], begin=[0], end=[width],
                ),
                matchings[matmul2]: R.strided_slice(
                    merged, axes=[2], begin=[width], end=[width * 2],
                ),
                matchings[matmul3]: R.strided_slice(
                    merged, axes=[2], begin=[width * 2], end=[width * 3],
                ),
            }

        new_func = rewrite_bindings(ctx, rewriter, func)


Declarative Rewriting with ``@R.rewriter``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For straightforward one-to-one replacements you can declare the pattern and
its replacement as two Relax functions in a single ``IRModule``.  The
``@R.rewriter`` decorator turns the module into a ``PatternMatchingRewriter``
object that can be applied directly.

.. code:: python

    from tvm.relax import script as R

    @R.rewriter
    class RewriteAddToPackedCall:
        @R.function
        def pattern(
            A: R.Tensor([16], "float32"),
            B: R.Tensor([16], "float32"),
        ):
            C = R.add(A, B)
            return C

        @R.function
        def replacement(
            A: R.Tensor([16], "float32"),
            B: R.Tensor([16], "float32"),
        ):
            C = R.call_pure_packed(
                "my_fast_add",
                A,
                B,
                ty_args=R.Tensor([16], "float32"),
            )
            return C

    # Apply to an IRModule or a single function
    rewritten_mod = RewriteAddToPackedCall(mod)

Composing Rewriters
~~~~~~~~~~~~~~~~~~~
Multiple ``PatternMatchingRewriter`` objects can be combined with the ``|``
operator so they run as a single pass:

.. code:: python

    combined = rewriter_a | rewriter_b
    result = combined(mod)

The left-hand rewriter is tried first; the right-hand rewriter only applies to
bindings that were **not** already modified by the left.


Using DPL in Compiler Passes
-----------------------------
The most common way DPL appears in the TVM codebase is through the
``FuseOpsByPattern`` pass, which uses ``FusionPattern`` objects to drive
operator fusion.

``FusionPattern``
~~~~~~~~~~~~~~~~~
A ``FusionPattern`` bundles four pieces of information:

- ``name`` -- a string label (e.g., ``"cutlass.matmul"``).
- ``pattern`` -- a ``DFPattern`` that describes the sub-graph to match.
- ``annotation_patterns`` -- a ``dict[str, DFPattern]`` that names interesting
  sub-patterns so the check function can inspect them.
- ``check`` -- an optional ``Callable[[PatternCheckContext], bool]`` that
  performs additional validation after a structural match succeeds.

.. code:: python

    from tvm.relax.dpl import wildcard, is_op
    from tvm.relax.transform import FusionPattern

    x = wildcard()
    w = wildcard()
    matmul = is_op("relax.matmul")(x, w)
    bias = wildcard()
    add = is_op("relax.add")(matmul, bias)

    pattern = FusionPattern(
        name="my_backend.matmul_bias",
        pattern=add,
        annotation_patterns={"matmul": matmul, "bias": bias, "lhs": x, "rhs": w},
        check=my_check_fn,
    )

``PatternCheckContext``
~~~~~~~~~~~~~~~~~~~~~~~
When ``FuseOpsByPattern`` finds a structural match, it calls the ``check``
function with a ``PatternCheckContext`` that provides:

- ``matched_expr`` -- the root expression of the match.
- ``annotated_expr`` -- a ``dict[str, Expr]`` resolved from the
  ``annotation_patterns``.
- ``matched_bindings`` -- a ``dict[Var, Expr]`` of bindings being fused.
- ``var_usages`` -- a ``dict[Var, Sequence[Var]]`` of variable use chains.
- ``value_to_bound_var`` -- a ``dict[Expr, Var]`` mapping values back to
  their bound variables.

Use the check function to enforce constraints that cannot be expressed
structurally (dtype restrictions, shape compatibility, attribute values, etc.):

.. code:: python

    from tvm.relax.transform import PatternCheckContext

    def my_check_fn(ctx: PatternCheckContext) -> bool:
        matmul_expr = ctx.annotated_expr["matmul"]
        # Only accept float16 output
        if matmul_expr.ty.dtype != "float16":
            return False
        return True

``FuseOpsByPattern``
~~~~~~~~~~~~~~~~~~~~
``FuseOpsByPattern`` is a module-level pass that takes a list of
``FusionPattern`` (or equivalent tuples) and groups every match into a fused
sub-function.

.. code:: python

    from tvm.relax.dpl import wildcard, is_op
    from tvm.relax.transform import FuseOpsByPattern

    # 1. Define the pattern
    w = wildcard()
    x = wildcard()
    wT = is_op("relax.permute_dims")(w)
    o = is_op("relax.matmul")(x, wT)
    annotations = {"o": o, "w": w, "x": x, "wT": wT}

    def check(ctx):
        transpose_call = ctx.annotated_expr["wT"]
        ndim = transpose_call.args[0].ty.ndim
        if ndim == -1:
            return False
        if ndim == 2 and transpose_call.attrs.axes is None:
            return True
        axes = list(range(ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return list(transpose_call.attrs.axes) == axes

    # 2. Run the pass
    mod = FuseOpsByPattern(
        [("transpose_matmul_fuse", o, annotations, check)],
        bind_constants=False,
    )(mod)

When ``annotate_codegen=True``, each fused function is additionally wrapped
with ``Codegen`` and ``global_symbol`` attributes, which is how backends like
CUTLASS and cuBLAS register themselves for external code generation.


Quick Reference
---------------

**Pattern construction**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - API
     - Description
   * - ``wildcard()``
     - Match any expression
   * - ``is_op(op_name)``
     - Match a Relax operator by name
   * - ``is_const()``
     - Match any constant
   * - ``is_var(name)`` / ``is_dfv(name)`` / ``is_gv(name)``
     - Match ``Var`` / ``DataflowVar`` / ``GlobalVar``
   * - ``is_tuple(fields)``
     - Match a tuple with given field patterns
   * - ``is_call_tir(name, args)``
     - Match ``R.call_tir``
   * - ``is_call_dps_packed(name, args)``
     - Match ``R.call_dps_packed``
   * - ``is_call_packed(name, args)``
     - Match ``R.call_packed``
   * - ``make_fused_bias_activation_pattern(...)``
     - Build ``op + bias + activation`` chain
   * - ``.has_dtype()`` / ``.has_shape()`` / ``.has_attr()`` / ``.has_ty()``
     - Attach constraints
   * - ``|`` / ``&`` / ``~``
     - Or / And / Not combinators
   * - ``^`` / ``>>``
     - used_by / only_used_by (sequence)

**Matching and rewriting**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - API
     - Description
   * - ``pattern.match(expr)``
     - Returns ``True`` if pattern matches
   * - ``pattern.extract_matched_expr(expr)``
     - Returns ``dict[DFPattern, Expr]`` or ``None``
   * - ``rewrite_call(pattern, rewriter, func)``
     - Rewrite individual expressions
   * - ``rewrite_bindings(ctx, rewriter, func)``
     - Rewrite across bindings in a ``DataflowBlock``
   * - ``PatternMatchingRewriter.from_module(mod)``
     - Declarative rewriter from ``IRModule``
   * - ``@R.rewriter``
     - Decorator shorthand for ``from_module``

**Pass integration**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - API
     - Description
   * - ``FusionPattern(name, pattern, annotations, check)``
     - Bundle pattern with metadata for ``FuseOpsByPattern``
   * - ``PatternCheckContext``
     - Runtime context passed to check functions
   * - ``FuseOpsByPattern(patterns, ...)``
     - Module pass that fuses matched sub-graphs
