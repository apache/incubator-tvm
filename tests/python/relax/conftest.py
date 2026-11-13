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

import pytest

import tvm_tirx
from tvm.relax.ir.instrument import WellFormedInstrument


@pytest.fixture
def unit_test_marks(request):
    """Get all marks applied to a test

    From https://stackoverflow.com/a/61379477.
    """
    marks = [m.name for m in request.node.iter_markers()]
    if request.node.parent:
        marks += [m.name for m in request.node.parent.iter_markers()]
    yield marks


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        (
            "skip_well_formed_check_before_transform: "
            "Suppress the default well-formed check before a IRModule transform"
        ),
    )
    config.addinivalue_line(
        "markers",
        (
            "skip_well_formed_check_after_transform: "
            "Suppress the default well-formed check after a IRModule transform"
        ),
    )


# By default, apply the well-formed check before and after all
# transforms.  Checking well-formed-ness after the transform ensures
# that all transforms produce well-formed output.  Checking
# well-formed-ness before the transform ensures that test cases
# (usually hand-written) are providing well-formed inputs.
#
# This is provided as a test fixture so that it can be overridden for
# specific tests.  If a test must provide ill-formed input to a
# transform, it can be marked with
# `@pytest.mark.skip_well_formed_check_before_transform`
@pytest.fixture(autouse=True)
def apply_instrument_well_formed(unit_test_marks):
    validate_before_transform = "skip_well_formed_check_before_transform" not in unit_test_marks
    validate_after_transform = "skip_well_formed_check_after_transform" not in unit_test_marks

    current = tvm_tirx.ir.transform.PassContext.current()
    instruments = list(current.instruments)

    if validate_before_transform or validate_after_transform:
        instruments.append(
            WellFormedInstrument(validate_before_transform=validate_before_transform)
        )

    override = tvm_tirx.ir.transform.PassContext(
        # With the new WellFormedInstrument appended
        instruments=instruments,
        # Forward all other parameters
        opt_level=current.opt_level,
        required_pass=current.required_pass,
        disabled_pass=current.disabled_pass,
        config=current.config,
    )
    with override:
        yield
