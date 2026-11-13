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
# pylint: disable=no-else-return, invalid-name
"""Developer API of add/remove/replace bindings in Relax."""

import tvm_ffi

import tvm_tirx

from . import Binding, DataflowBlock, Expr, Function, Var, _ffi_api


@tvm_ffi.register_object("relax.DataflowBlockRewrite")
class DataflowBlockRewrite(tvm_ffi.Object):
    """
    A binding/statement-level dataflow block rewriter.

    Notes
    -----
    Due to the immutable and copy-on-write nature of TVM AST nodes, the rewriting is not done in
    place. Instead, a new DataflowBlock is created and returned with mutated_dfb. Similarly, its new
    root Function is created and returned by mutated_root_fn. To apply this change for an IRModule,
    use mutate_irmodule which rewrites the old function that registered in the constructor.
    """

    __slots__ = ("__dict__",)

    def __init__(self, dfb: DataflowBlock, root_fn: Function):
        """
        Construct a rewriter with the DataflowBlock to rewrite and its root function.

        Parameters
        ----------
        dfb : DataflowBlock
            The DataflowBlock to rewrite.
        root_fn : Function
            The root function of the DataflowBlock.
        """
        self.func_name = root_fn.__name__ if hasattr(root_fn, "__name__") else None
        self.__init_handle_by_constructor__(
            _ffi_api.DataflowBlockRewrite,
            dfb,
            root_fn,  # type: ignore
        )

    def replace_all_uses(self, old_var: Var, new_var: Var) -> None:
        """
        Replace all uses of old_var with new_var.

        Parameters
        ----------
        old_var : Var
            The old variable to replace.
        new_var : Var
            The new variable to replace with.
        """
        _ffi_api.dfb_rewrite_replace_all_uses(self, old_var, new_var)  # type: ignore

    def add_binding(self, binding: Binding) -> None:
        return _ffi_api.dfb_rewrite_add_binding(self, binding)  # type: ignore

    def add(self, expr: Expr, name: str | None = None, is_dfvar: bool = False) -> None:
        """
        Add a new statement to the DataflowBlock with an automatically generated variable name.

        Parameters
        ----------
        expr : Expr
            The expression to add.
        name : Optional[str], optional
            Variable name, by default None
        is_dfvar : bool, optional
            The variable type, by default False

        Notes
        -----
        If the variable name is not given, it will be automatically generated in a form of
        "tmp${COUNTER}". The variable type will be DataflowVar if is_dfvar is True, otherwise
        it will be Var. Being Var means the variables are output variables of the DataflowBlock.
        While being DataflowVar means the variables are internal variables of the DataflowBlock.
        """
        _ffi_api.dfb_rewrite_add(self, expr, name, is_dfvar)  # type: ignore

    def remove_unused(self, var: Var, allow_undef=False) -> None:
        """
        Remove a statement by its variable definition if and only if it is unused.

        Parameters
        ----------
        var : Var
            The unused variable definition.
        allow_undef : bool, optional
            Whether to allow var being undefined variable, by default False

        Raises
        ------
        RuntimeError if the variable is used or undefined (allow_undef=False).
        """
        _ffi_api.dfb_rewrite_remove_unused(self, var, allow_undef)  # type: ignore

    def remove_all_unused(self) -> None:
        """
        Remove all unused variables.

        Notes
        -----
        This could remove unused variables in other DataflowBlocks as well.
        """
        _ffi_api.dfb_rewrite_remove_all_unused(self)  # type: ignore

    def mutated_dfb(self) -> DataflowBlock:
        """
        Returns the mutated DataflowBlock.
        """
        return self.dfb

    def mutated_root_fn(self) -> Function:
        """
        Returns the mutated root function.
        """
        ret = self.root_fn
        if self.func_name:
            ret.__name__ = self.func_name
        return ret

    def mutate_irmodule(self, irmodule: tvm_tirx.IRModule) -> tvm_tirx.IRModule:
        """
        Return an updated IRModule by replacing the old function with the mutated root function.

        Parameters
        ----------
        irmodule : tvm_tirx.IRModule
            The base IRModule to update.

        Returns
        -------
        tvm_tirx.IRModule
            The updated IRModule.
        """
        ret = _ffi_api.dfb_rewrite_mutate_irmodule(self, irmodule)  # type: ignore
        if hasattr(irmodule, "__name__"):
            ret.__name__ = irmodule.__name__
        return ret
