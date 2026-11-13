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
"""A pass that attaches external modules to the IRModule.

Note: "external modules" here refers to `relax.frontend.nn.ExternModule`.
"""

from typing import TYPE_CHECKING

from tvm_tirx.ir import IRModule
from tvm_tirx.ir.transform import PassContext, module_pass

if TYPE_CHECKING:
    from tvm.relax.frontend.nn import ExternModule


@module_pass(opt_level=0, name="AttachExternalModules")
class AttachExternModules:  # pylint: disable=too-few-public-methods
    """Attach variable bounds to each Relax function, which primarily helps with memory planning."""

    def __init__(self, extern_modules: list["ExternModule"]):
        self.extern_modules = extern_modules

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        """Entrypoint"""
        from tvm.relax.frontend.nn import (  # pylint: disable=import-outside-toplevel
            ExternModule,
        )

        def _load(ext_mod: ExternModule):
            assert isinstance(ext_mod, ExternModule), f"Expected ExternModule, but got: {ext_mod}"
            return ext_mod.load()

        mod_attrs = dict(mod.attrs) if mod.attrs else {}
        external_mods = mod_attrs.get("external_mods", [])
        for ext in self.extern_modules:
            external_mods.append(_load(ext))
        mod = mod.with_attr("external_mods", external_mods)
        return mod
