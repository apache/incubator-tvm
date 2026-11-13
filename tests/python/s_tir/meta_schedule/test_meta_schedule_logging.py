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
"""Test MetaSchedule logger initialization with application-installed handlers."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("handler_type", ["none", "stream", "null"])
def test_get_loggers_from_work_dir_existing_handlers(tmp_path, handler_type):
    # dictConfig changes process-global logging state, so isolate it from pytest.
    program = """
import logging
import sys
from pathlib import Path

from tvm.s_tir.meta_schedule.logging import get_loggers_from_work_dir

handler_type = sys.argv[2]
if handler_type != "none":
    handler = logging.StreamHandler() if handler_type == "stream" else logging.NullHandler()
    assert handler.get_name() is None
    logging.getLogger("tvm.s_tir.meta_schedule").addHandler(handler)

for iteration in range(2):
    work_dir = Path(sys.argv[1]) / str(iteration)
    loggers = get_loggers_from_work_dir(str(work_dir), ["main", "second"])
    assert len(loggers) == 2
    for index, logger in enumerate(loggers):
        message = f"task-{index}-iteration-{iteration}"
        logger.info(message)
        log_file = work_dir / "logs" / f"{logger.name}.log"
        assert message in log_file.read_text()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), handler_type],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
