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

# pylint: disable=invalid-name
"""The TIR backend compilation pipeline for Adreno"""

from tvm_tirx.tirx import compilation_pipeline as tir_pipeline


def default_tir_pipeline(*, prepare_only=False, target=None):
    """Lower S-TIR texture operations before target classification."""
    from tvm.s_tir.pipeline import _s_tir_pipeline

    return _s_tir_pipeline(prepare_only=prepare_only, target=target, adreno=True)


tir_pipeline.PIPELINE_MAP["adreno"] = default_tir_pipeline
