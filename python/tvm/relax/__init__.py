# isort: skip_file
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, wrong-import-position
"""The Relax IR namespace containing the IR, type, operator, builder, vm, etc."""

from tvm.runtime import vm
from tvm.runtime.vm import VirtualMachine, VMInstrumentReturnKind
from tvm_tirx.ir import Call

# Global information
from .global_info import DummyGlobalInfo, VDevice

# Expr
from .expr import (
    Expr,
    Location,
    GlobalVar,
    Var,
    DataflowVar,
    Binding,
    MatchCast,
    VarBinding,
    BindingBlock,
    DataflowBlock,
    SeqExpr,
    ShapeExpr,
    Tuple,
    TupleGetItem,
    Function,
    ExternFunc,
    IfExpr,
    prim_value,
)

from .expr import const, extern, get_shape_of

# Type
from .type import (
    Type,
    AnyType,
    ObjectType,
    ShapeType,
    TensorType,
    TupleType,
    FuncType,
    PackedFuncType,
)

# VM
from .exec_builder import ExecBuilder

# Operator
from .op.base import (
    call_tir,
    call_tir_inplace,
    call_tir_packed,
    call_pure_packed,
    call_dps_packed,
    call_tir_with_grad,
)

# BlockBuilder
from .block_builder import BlockBuilder

# ExprFunctor
from .expr_functor import ExprFunctor, PyExprVisitor, PyExprMutator

# pipeline
from .pipeline import get_default_pipeline
from .pipeline import get_pipeline
from .pipeline import register_pipeline

# utils
from .utils import convert_to_expr

# BasePyModule
from .base_py_module import BasePyModule

# Import submodules in the last to avoid dependency
from . import exec_builder
from . import expr
from . import ty
from . import type
from . import analysis
from . import transform
from . import block_builder
from . import op
from . import backend
from . import training
from . import distributed
from . import frontend
from . import utils

# VM
from .vm_build import build, VMExecutable

from .binding_rewrite import DataflowBlockRewrite

import tvm_tirx.script
import tvm

tvm_tirx.script.register_dialect(
    "relax", "tvm.relax.script", builder_path="tvm.relax.script.ir_builder"
)


def _initialize_script_namespace() -> None:
    from . import script

    script._initialize()


from tvm_tirx.script.parser import register_namespace_initializer as _register_namespace_initializer

_register_namespace_initializer(_initialize_script_namespace, aliases=("relax",))

# Register the exported operator printer names for compiler diagnostics as well.
from .script import ir_builder as _script_ir_builder

from .script import printer as _script_printer
