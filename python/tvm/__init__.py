# isort: skip_file
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
# pylint: disable=redefined-builtin, wildcard-import
"""TVM compiler dialects and runtime extensions built on TIRx."""

import os
import sys

from typing import TYPE_CHECKING


def _is_config_mode() -> bool:
    """Recognize the config CLI before loading Python or native TVM components."""
    if os.path.basename(sys.argv[0]) in ("tvm-config", "tvm-config.exe"):
        return True
    argv = getattr(sys, "orig_argv", [])
    return any(arg == "-m" and argv[i + 1 : i + 2] == ["tvm.config"] for i, arg in enumerate(argv))


if TYPE_CHECKING or not _is_config_mode():
    from .libinfo import __version__
    from .base import _RUNTIME_ONLY
    from . import runtime
    from . import rpc

    if not _RUNTIME_ONLY:
        from . import te
        from . import s_tir
        from .driver import compile
        from . import relax
