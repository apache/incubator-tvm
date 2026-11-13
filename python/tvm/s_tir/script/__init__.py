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
"""S-TIR construction namespace, exposed as ``tvm_tirx.script.s_tir``."""

import importlib as _importlib
import sys as _sys

_initialized = False
_initializing = False


def _initialize():
    global _initialized, _initializing
    if _initialized or _initializing:
        return
    _initializing = True
    try:
        from tvm_tirx.script.parser import entry, protocol_registry, register_namespace
        from tvm_tirx.tirx import script as shared

        from . import ir_builder as builder

        shared._initialize()
        globals().update(
            (name, getattr(shared, name))
            for name in shared.__all__
            if name not in ("builder", "ir_builder", "jit")
        )
        for name in (
            "CommReducer",
            "Reduce",
            "comm_reducer",
            "async_copy_scope",
            "async_commit",
            "async_wait",
            "manual_sync",
            "function_",
            "arg_",
            "bind_",
            "check_well_formed_",
            "sblock",
            "match_tensor",
            "init",
            "where",
            "reads",
            "writes",
            "sblock_attr",
            "sblock_alloc_tensor",
            "axis",
            "block_name_suffix_context",
        ):
            globals()[name] = getattr(builder, name)
        # The operations are shared, but syntax is keyed by the registered namespace.
        for table in (
            protocol_registry.SCALAR_ANNOTATION_DTYPE,
            protocol_registry.MUTABLE_CELL_DECL,
        ):
            table.update(
                {
                    "s_tir." + key[len("tirx.") :]: value
                    for key, value in list(table.items())
                    if key.startswith("tirx.")
                }
            )
        globals().update(
            function=entry.make_decorator(builder, namespace_path="s_tir.function"),
            inline=entry.make_macro_decorator(
                builder, namespace_path="s_tir.inline", preserve_return=True, late_binding=True
            ),
            macro=entry.make_macro_decorator(
                builder, namespace_path="s_tir.macro", preserve_return=False
            ),
        )
        register_namespace("s_tir", _sys.modules[__name__])
        globals()["__all__"] = sorted(name for name in globals() if not name.startswith("_"))
        _initialized = True
    finally:
        _initializing = False


def __getattr__(name):
    if name == "ir_builder":
        return _importlib.import_module(__name__ + ".ir_builder")
    if name.startswith("_") and name != "__all__":
        raise AttributeError(name)
    _initialize()
    if name in globals():
        return globals()[name]
    raise AttributeError(name)
