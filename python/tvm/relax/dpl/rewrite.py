# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""APIs for pattern-based rewriting."""

from collections.abc import Callable

import tvm_ffi
from tvm_ffi import register_object

from tvm_tirx.ir import IRModule

from ..expr import Expr, Function, Var
from . import _ffi as ffi
from .context import PatternContext
from .pattern import DFPattern


@register_object("relax.dpl.PatternMatchingRewriter")
class PatternMatchingRewriter(tvm_ffi.Object):
    """A pattern-matching rewriter for Relax"""

    @staticmethod
    def from_pattern(
        pattern: DFPattern,
        func: Callable[[Expr, dict[DFPattern, Expr]], Expr],
    ) -> "PatternMatchingRewriter":
        """Construct from a pattern and rewriter-function

        The replacements performed by the rewriter will be equivalent
        to using the `pattern` and `func` as arguments to
        `rewrite_call`.

        Parameters
        ----------
        pattern: DFPattern

            The pattern to be matched against.

        func: Callable[[Expr, Dict[DFPattern, Expr]], Expr]

            A function that returns the rewritten expression.  See
            `rewrite_call` for details and examples.


        Returns
        -------
        rewriter_obj: PatternMatchingRewriter

            The rewriter object

        """
        return ffi.PatternMatchingRewriterFromPattern(
            pattern,
            func,
        )  # type: ignore

    @staticmethod
    def from_module(mod: IRModule) -> "PatternMatchingRewriter":
        """Construct a rewriter from an IRModule

        The IRModule must have two publicly-exposed functions,
        `pattern` and `replacement`, where `pattern` and `replacement`
        have the same function signature, as shown in the example
        below.

        .. code-block:: python

            @I.ir_module
            class RewriteAddIntoMultiply:
                @R.function
                def pattern(A: R.Tensor):
                    B = A + A
                    return B

                @R.function
                def replacement(A: R.Tensor):
                    B = A * 2
                    return B

            rewriter = PatternMatchingRewriter.from_module(RewriteAddIntoMultiply)
            rewritten_ir_module = rewriter(ir_module)

        To support the common case of defining an IRModule with
        TVMScript, then immediately turning it into a rewriter, the
        `@R.rewriter` annotation can be used.

        .. code-block:: python

            @R.rewriter
            class RewriteAddIntoMultiply:
                @R.function
                def pattern(A: R.Tensor):
                    B = A + A
                    return B

                @R.function
                def replacement(A: R.Tensor):
                    B = A * 2
                    return B

            rewritten_ir_module = RewriteAddIntoMultiply(ir_module)

        Parameters
        ----------
        mod: IRModule

            A module with `pattern` and `replacement` functions,
            defining a rewrite rule.


        Returns
        -------
        rewriter_obj: PatternMatchingRewriter

            The rewriter object

        """
        return ffi.PatternMatchingRewriterFromModule(mod)  # type: ignore

    def __call__(self, obj: Expr | IRModule) -> Expr | IRModule:
        """Apply the rewriter

        Parameters
        ----------
        obj: Union[Expr, IRModule])

            The object to be rewritten.  May be applied to either a
            relax expression, or an IRModule.

        Returns
        -------
        updated: Union[Expr, IRModule]

            The rewritten object

        """
        return ffi.PatternMatchingRewriterApply(self, obj)

    def __or__(self, other: "PatternMatchingRewriter") -> "PatternMatchingRewriter":
        """Compose two rewriters

        Composing two rewrite rules together allows them to be applied
        in a single Relax-level transformation.

        Parameters
        ----------
        other: PatternMatchingRewriter

            Another rewrite rule

        Returns
        -------
        PatternMatchingRewriter

            A rewriter that will apply either rewrite pattern

        """
        return OrRewriter(self, other)


@register_object("relax.dpl.ExprPatternRewriter")
class ExprPatternRewriter(PatternMatchingRewriter):
    def __init__(self, pattern, func):
        self.__init_handle_by_constructor__(
            ffi.PatternRewriter,
            pattern,
            func,
        )  # type: ignore


@register_object("relax.dpl.OrRewriter")
class OrRewriter(PatternMatchingRewriter):
    def __init__(self, lhs, rhs):
        self.__init_handle_by_constructor__(
            ffi.OrRewriter,
            lhs,
            rhs,
        )  # type: ignore


@register_object("relax.dpl.TupleRewriter")
class TupleRewriter(PatternMatchingRewriter):
    def __init__(self, patterns, func):
        self.__init_handle_by_constructor__(
            ffi.TupleRewriter,
            patterns,
            func,
        )  # type: ignore


def rewrite_call(
    pattern: DFPattern,
    rewriter: Callable[[Expr, dict[DFPattern, Expr]], Expr],
    func: Function,
) -> Function:
    """
    Rewrite a function with the given pattern and the rewriter function.

    Parameters
    ----------
    pattern: DFPattern
        The pattern to match.

    rewriter: Callable[[Expr, Dict[DFPattern, Expr]], Expr]
        The function to be called on a successful matching for rewriting. Given the matched
        call node and the map of patterns and matched expressions, it should return a new call node
        to replace the original one or the original matched call node as is.

        For example, to replace x + x with 2 * x, we can write the rewriter as follows:
        ```
        x = wildcard()
        pattern = is_op("relax.add")(x, x)

        def rewriter(orig, matchings):
            return R.multiply(matchings[x], R.const(2, "float32"))
        ```

    func: Function
        The function to rewrite.

    Returns
    -------
    rewritten_func: Function
        The rewritten or the input function, depending on the pattern matching result.
    """
    return ffi.rewrite_call(pattern, rewriter, func)


def rewrite_bindings(
    ctx: PatternContext,
    rewriter: Callable[[dict[DFPattern, Var], dict[Var, Expr]], dict[Var, Expr]],
    func: Function,
) -> Function:
    """
    Rewrite a function with the given pattern and the rewriter function.

    Parameters
    ----------
    ctx: PatternContext
        The pattern constraint context under which rewriting takes place.

    rewriter: Callable[[Dict[DFPattern, Var], Dict[Var, Expr]], Dict[Var, Expr]]
        The function to be called on a successful matching for rewriting. Given the map of patterns
        and corresponding variables (bound variables or parameters), it should return a map that
        specifies new values for matched bound variables. It can refer to the passed bindings to
        create the replacement expressions.

        For example, to rewrite three matmuls for QKV projection in transformer models into one
        matmul followed by slicing, one can use the follwoing rewriter:
        ```
        inp_pat = wildcard()
        Q_weight_pat, K_weight_pat, V_weight_pat = wildcard(), wildcard(), wildcard()

        matmul1 = is_op("relax.matmul")(inp_pat, Q_weight_pat)
        matmul2 = is_op("relax.matmul")(inp_pat, K_weight_pat)
        matmul3 = is_op("relax.matmul")(inp_pat, V_weight_pat)

        def rewriter(matchings):
            inp = matchings[inp_pat]
            Q_weight = matchings[Q_weight_pat]
            K_weight = matchings[K_weight_pat]
            V_weight = matchings[V_weight_pat]
            width = Q_weight.ty.shape[1]

            concat = R.concat([Q_weight, K_weight, V_weight], axis=1)
            matmul = R.matmul(inp, concat)
            Q = R.strided_slice(matmul, axes=[2], begin=[0], end=[width])
            K = R.strided_slice(matmul, axes=[2], begin=[width], end=[width * 2])
            V = R.strided_slice(matmul, axes=[2], begin=[width * 2], end=[width * 3])

            # matchings[matmul1] gives the bound variable in the binding whose RHS matches with
            # the matmul1 pattern. For example, lv0 in lv0 = R.matmul(x1, w0).
            # We want to replace the RHS of this binding with Q.
            return {matchings[matmul1]: Q, matchings[matmul2]: K, matchings[matmul3]: V}
        ```

    func: Function
        The function to rewrite.

    Returns
    -------
    rewritten_func: Function
        The rewritten or the input function, depending on the pattern matching result.
    """
    return ffi.rewrite_bindings(ctx, rewriter, func)
