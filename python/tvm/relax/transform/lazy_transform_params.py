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
# pylint: disable=invalid-name, unused-argument, missing-function-docstring, abstract-method, missing-class-docstring
# ruff: noqa: RUF005
"""Relax LazyTransformParams pass."""

import tvm_tirx
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator, visitor
from tvm_tirx import IRModule


@visitor
class ForwardCollector(PyExprVisitor):
    """
    Perform a forward pass to collect the following information:
    out_tuple_map: map from var to its index in the output tuple
    var_tuple_get_item: list of var that is bound to v = params[i]

    Parameters
    ----------
    tuple_var: relax.Var
        The output tuple var

    input_params: relax.Var
        The input tuple var

    """

    def __init__(self, tuple_var: relax.Var, input_params: relax.Var) -> None:
        self.out_tuple_map = {}
        self.out_tuple_var = tuple_var
        self.input_params = input_params
        self.var_tuple_get_item = []
        self.is_tuple_get_item_input = False

    def visit_tuple_getitem_(self, op: relax.TupleGetItem) -> None:
        if op.tuple_value == self.input_params:
            self.is_tuple_get_item_input = True
        else:
            self.is_tuple_get_item_input = False
        super().visit_tuple_getitem_(op)

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if binding.var == self.out_tuple_var:
            assert isinstance(binding.value, relax.Tuple)
            for i, expr in enumerate(binding.value.fields):
                if expr not in self.out_tuple_map:
                    self.out_tuple_map[expr] = []
                self.out_tuple_map[expr].append(relax.prim_value(i))
        else:
            self.is_tuple_get_item_input = False
            super().visit_var_binding_(binding)
            if self.is_tuple_get_item_input:
                self.var_tuple_get_item.append(binding.var)


@visitor
class LivenessAnalysis(PyExprVisitor):
    """
    Perform a backward pass to collect the following information:
    var_liveness_end: map from var to the list of var whose liveness is killed by this var binding

    Parameters
    ----------
    out_tuple_var: relax.Var
        The output tuple var
    input_params: set
        The set of vars that are bound to v = params[i]
    """

    def __init__(self, out_tuple_var: relax.Var) -> None:
        self.last_appear_in_var_binding = []
        self.out_tuple_var = out_tuple_var
        self.var_liveness_end = {}
        self.ended_vars = set()

    def visit_binding_block_(self, block: relax.BindingBlock) -> None:
        for binding in reversed(block.bindings):
            self.visit_binding(binding)

    def visit_var_(self, op: relax.Var) -> None:
        if op not in self.ended_vars:
            self.last_appear_in_var_binding.append(op)
            self.ended_vars.add(op)

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if self.out_tuple_var == binding.var:
            return
        self.last_appear_in_var_binding = []
        super().visit_var_binding_(binding)
        # param[i] is in output
        if binding.var not in self.ended_vars:
            self.last_appear_in_var_binding.append(binding.var)
            self.ended_vars.add(binding.var)
        self.var_liveness_end[binding.var] = self.last_appear_in_var_binding


class LazyTransformParamsFuncCreator:
    """
    Transform transform_params functions into a lazy version.

    Parameters
    ----------
    mod: IRModule
        The module to be transformed
    """

    def __init__(
        self,
        fget_item,
        fset_item,
        extra_get_item_params,
        extra_set_item_params,
        mod: IRModule = None,
    ) -> None:
        self.mod = mod
        self.fget_item = fget_item
        self.extra_get_item_params = extra_get_item_params
        self.fset_item = fset_item
        self.extra_set_item_params = extra_set_item_params
        self.input_params_set = None
        self.out_tuple_map = None
        self.out_tuple_var = None
        self.memory_free_insertion = None

    def transform(self, func: relax.Function) -> relax.Function:
        if "num_input" in func.attrs:
            num_input = func.attrs["num_input"]
        else:
            num_input = 0

        seq_expr = func.body
        self.out_tuple_var = seq_expr.body

        # Step 1. collect out_tuple_map and input_params_set
        forward_collector = ForwardCollector(self.out_tuple_var, func.params[num_input])
        forward_collector.visit_expr(func)
        self.out_tuple_map = forward_collector.out_tuple_map
        # input_params_set is the set of binding var for var = params[i]
        self.input_params_set = set(forward_collector.var_tuple_get_item)

        # Step 2. liveness analysis and get where to insert kill_object instruction
        liveness = LivenessAnalysis(self.out_tuple_var)
        liveness.visit_expr(func)
        self.memory_free_insertion = liveness.var_liveness_end

        # Step 3. rewrite get item and set item
        if self.fget_item is not None:
            new_func = LazyInputMutator(self, self.mod).visit_expr(func)

        new_body = new_func.body
        if self.fset_item is not None:
            # The LazyOutputMutator only inspects variable bindings
            # for replacement.  If the output tuple includes elements
            # that do not have a variable binding, such as
            # `relax.Const`, these must still produce a call to the
            # `"set_item"` function.
            leaf_outputs = {
                expr: indices
                for expr, indices in self.out_tuple_map.items()
                if not isinstance(expr, relax.Var)
            }
            if leaf_outputs:
                new_bindings = [
                    relax.VarBinding(
                        relax.Var("_", relax.AnyType()),
                        relax.Call(
                            relax.ExternFunc(self.fset_item),
                            [*self.extra_set_item_params, index, expr],
                            None,
                            [relax.AnyType()],
                        ),
                    )
                    for expr, indices in leaf_outputs.items()
                    for index in indices
                ]
                new_body = relax.SeqExpr(
                    [*new_body.blocks, relax.BindingBlock(new_bindings)], new_body.body
                )

            new_body = LazyOutputMutator(self, self.mod).visit_expr(new_body)

        # Step 4. Add parameters of get_item and set_item (except index) to the function.
        params = [
            *func.params[:num_input],
            *self.extra_get_item_params,
            *self.extra_set_item_params,
        ]

        # Step 5. Find all shape parameters that should be retained as
        # parameters.
        symbolic_vars = relax.analysis.defined_symbolic_vars(func)
        if symbolic_vars:

            def unpack_ty(ty):
                if isinstance(ty, relax.TupleType):
                    for field in ty.fields:
                        yield from unpack_ty(field)
                else:
                    yield ty

            # direct iterate over the type annotation
            for param in func.params[num_input:]:
                for ty in unpack_ty(param.ty):
                    if isinstance(ty, tvm_tirx.ir.PrimType | relax.ShapeType):
                        params.append(relax.Var("symbolic_var_holder", ty))

        return relax.Function(
            params,
            new_body,
            relax.AnyType(),
            attrs=func.attrs,
            is_pure=False,
        ).without_attr("relax.force_pure")


@mutator
class LazyInputMutator(PyExprMutator):
    def __init__(self, func_creator, mod: IRModule | None = None) -> None:
        self.func_creator = func_creator
        super().__init__(mod)

    def visit_function_(self, func: relax.Function) -> relax.Expr:
        if "num_input" in func.attrs:
            num_input = func.attrs["num_input"]
        else:
            num_input = 0

        params = list(func.params)[num_input:]
        if len(params) == 1 and isinstance(params[0].ty, relax.TupleType):
            self.tuple_param = params[0]
            self.params = {}
        else:
            self.tuple_param = None
            self.params = {var: i for i, var in enumerate(params)}
        func = relax.Function(
            func.params[:num_input],
            func.body,
            func.ret_ty,
            is_pure=False,
            attrs=func.attrs,
            loc=func.loc,
        ).without_attr("relax.force_pure")
        output = super().visit_function_(func)
        self.tuple_param = None
        self.params = {}
        return output

    def visit_var_(self, var: relax.Var) -> relax.Expr:
        if var in self.params:
            index = self.params[var]
            get_item_result = self.builder_.emit(
                relax.Call(
                    relax.ExternFunc(self.func_creator.fget_item),
                    self.func_creator.extra_get_item_params + [relax.prim_value(index)],
                    None,
                    [relax.AnyType()],
                )
            )
            match_cast = relax.MatchCast(var, get_item_result, var.ty)
            self.builder_.emit_normalized(match_cast)

            del self.params[var]

        return super().visit_var_(var)

    def visit_tuple_getitem_(self, node: relax.TupleGetItem) -> relax.Expr:
        ty = node.ty

        node = super().visit_tuple_getitem_(node)

        if self.tuple_param is not None and node.tuple_value.same_as(self.tuple_param):
            get_item_result = self.builder_.emit(
                relax.Call(
                    relax.ExternFunc(self.func_creator.fget_item),
                    self.func_creator.extra_get_item_params + [relax.prim_value(node.index)],
                    None,
                    [relax.AnyType()],
                )
            )
            return self.builder_.match_cast(get_item_result, ty)
        else:
            return node


@mutator
class LazyOutputMutator(PyExprMutator):
    def __init__(self, func_creator, mod: IRModule | None = None) -> None:
        self.func_creator = func_creator
        self.killed_vars = set()
        super().__init__(mod)

    def visit_var_(self, var: relax.Var) -> None:
        assert var not in self.killed_vars
        return super().visit_var_(var)

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if binding.var == self.func_creator.out_tuple_var:
            # The function after rewriting returns a empty tuple.
            func_output = self.builder_.emit(relax.Tuple([]))
            self.set_var_remap(binding.var, func_output)
            return

        super().visit_var_binding_(binding)

        if binding.var in self.func_creator.memory_free_insertion:
            for var in self.func_creator.memory_free_insertion[binding.var]:
                if var in self.func_creator.out_tuple_map:
                    self.killed_vars.add(var)
                    for index in self.func_creator.out_tuple_map[var]:
                        # rewrite set item
                        self.builder_.emit(
                            relax.Call(
                                relax.ExternFunc(self.func_creator.fset_item),
                                self.func_creator.extra_set_item_params
                                + [index, super().visit_var_(var)],
                                None,
                                [relax.AnyType()],
                            ),
                            name_hint="_",
                        )

                if var in self.func_creator.input_params_set:
                    self.builder_.emit(
                        relax.op.vm.kill_object(super().visit_var_(var)), name_hint="_"
                    )


@tvm_tirx.ir.transform.module_pass(opt_level=0, name="LazyTransformParams")
class LazyTransformParams:
    """
    Convert transform_params functions into a lazy version.
    (Load the input to memory on demand, and immediately free it after the last use.)

    Note: ToNonDataflow() and RemovePurityTracking() should be invoked before this pass.

    Parameters
    ----------
    fget_item: str
        The name of the get_item function.
    fset_item: str
        The name of the set_item function.
    extra_get_item_params: list of relax.Var
        The parameters of the get_item function except index.
        The given parameters will be placed before index.
        For example, if extra_get_item_params is [param1, param2], then the pass will generate
        call_packed(fget_item, [param1, param2, index])
    extra_set_item_params: list of relax.Var
        The parameters of the set_item function except index and value.
        The given parameters will be placed before index and value.
        For example, if extra_set_item_params is [param1, param2], then the pass will generate
        call_packed(fset_item, [param1, param2, index, value])
    """

    def __init__(
        self,
        fget_item="get_item",
        fset_item="set_item",
        extra_get_item_params=None,
        extra_set_item_params=None,
    ) -> None:
        self.fget_item = fget_item
        self.extra_get_item_params = [] if extra_get_item_params is None else extra_get_item_params
        assert self.fget_item is not None, "transforming set_item only is not supported"
        self.fset_item = fset_item
        self.extra_set_item_params = [] if extra_set_item_params is None else extra_set_item_params

    def transform_module(self, mod: IRModule, ctx: tvm_tirx.ir.transform.PassContext) -> IRModule:
        lazy_mutator = LazyTransformParamsFuncCreator(
            self.fget_item,
            self.fset_item,
            self.extra_get_item_params,
            self.extra_set_item_params,
            mod,
        )
        builder = relax.BlockBuilder(mod)
        for gv, _ in mod.functions_items():
            if gv.name_hint.endswith("transform_params"):
                func = mod[gv]
                if not isinstance(func, relax.Function):
                    continue
                func = lazy_mutator.transform(func)
                builder.update_func(gv, func)

        return builder.get()
