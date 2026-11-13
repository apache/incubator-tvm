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
"""Relax-owned printing conveniences."""

import tvm_ffi
from tvm_ffi.access_path import AccessPath

from tvm_tirx.ir.base import register_dialect_script_method
from tvm_tirx.script.printer import PrinterConfig


def _relax_script(
    self,
    *,
    name: str | None = None,
    show_meta: bool = False,
    int_dtype: str = "int32",
    float_dtype: str = "void",
    verbose_expr: bool = False,
    indent_spaces: int = 4,
    print_line_numbers: bool = False,
    num_context_lines: int = -1,
    path_to_underline: list[AccessPath] | None = None,
    path_to_annotate: dict[AccessPath, str] | None = None,
    obj_to_underline: list[tvm_ffi.Object] | None = None,
    obj_to_annotate: dict[tvm_ffi.Object, str] | None = None,
    extra_config: dict | None = None,
) -> str:
    merged_extra = dict(extra_config or {})
    return tvm_ffi.get_global_func("script.printer.ReprPrintRelax")(
        self,
        PrinterConfig(
            name=name,
            show_meta=show_meta,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            verbose_expr=verbose_expr,
            indent_spaces=indent_spaces,
            print_line_numbers=print_line_numbers,
            num_context_lines=num_context_lines,
            path_to_underline=path_to_underline,
            path_to_annotate=path_to_annotate,
            obj_to_underline=obj_to_underline,
            obj_to_annotate=obj_to_annotate,
            extra_config=merged_extra,
        ),
    )


register_dialect_script_method("_relax_script", _relax_script)
