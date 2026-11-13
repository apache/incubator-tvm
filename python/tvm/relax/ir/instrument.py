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
"""Common relax pass instrumentation across IR variants."""

import tvm_tirx
from tvm import relax


@tvm_tirx.ir.transform.pass_instrument
class WellFormedInstrument:
    """An instrument that checks the input/output IRModule of the Pass
    is well formed. It will skip specific passes, like Normalize.

    Parameters
    ----------
    check_ty: bool

        If True, validate the type in the module.  If False,
        skip these checks.

    validate_before_transform: bool

        If True (default), perform a well-formed check before running
        a transform.  If False, only perform the well-formed check
        after running a transform.
    """

    def __init__(self, check_ty: bool = True, validate_before_transform: bool = True):
        self.skip_pass_name = ["Normalize", "NormalizeGlobalVar", "ResolveGlobals"]
        self.check_ty = check_ty
        self.validate_before_transform = validate_before_transform

    def run_before_pass(self, mod, pass_info):
        if self.validate_before_transform:
            self._check(mod, pass_info.name)

    def run_after_pass(self, mod, pass_info):
        self._check(mod, pass_info.name)

    def _check(self, mod, pass_name):
        if pass_name not in self.skip_pass_name:
            relax.analysis.check_well_formed(mod, self.check_ty)
