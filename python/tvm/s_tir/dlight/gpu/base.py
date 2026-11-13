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
"""Base schedule rule for GPU operators."""

from tvm_tirx.target import Target

from ..base import ScheduleRule


class GPUScheduleRule(ScheduleRule):  # pylint: disable=too-few-public-methods
    """The Schedule Rule specific to GPU targets, will return None if the target is not GPU."""

    def is_target_available(self, target: Target) -> bool:
        """Check whether the target is available for gpu rule.

        Parameters
        ----------
        target : Target
            The compilation target to check.

        Returns
        -------
        available : bool
            Whether the target is available for this rule.
        """
        return super().is_target_available(target) and "gpu" in target.keys
