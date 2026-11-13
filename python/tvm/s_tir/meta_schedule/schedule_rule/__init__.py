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
"""
The tvm.s_tir.meta_schedule.schedule_rule package.
Meta Schedule schedule rules are used for modification of
blocks in a schedule. See also PostOrderApply.
"""

from .add_rfactor import AddRFactor
from .apply_custom_rule import ApplyCustomRule
from .auto_bind import AutoBind
from .auto_inline import AutoInline, InlineConstantScalars
from .cross_thread_reduction import CrossThreadReduction
from .multi_level_tiling import (
    MultiLevelTiling,
    MultiLevelTilingWideVector,
    MultiLevelTilingWithIntrin,
    ReuseType,
)
from .parallel_vectorize_unroll import ParallelizeVectorizeUnroll
from .random_compute_location import RandomComputeLocation
from .schedule_rule import PyScheduleRule, ScheduleRule
