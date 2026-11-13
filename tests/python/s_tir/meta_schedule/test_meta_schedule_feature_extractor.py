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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

import numpy as np

import tvm_tirx.runtime
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.feature_extractor import PyFeatureExtractor
from tvm.s_tir.meta_schedule.search_strategy import MeasureCandidate
from tvm.s_tir.utils import derived_object


def test_meta_schedule_feature_extractor():
    @derived_object
    class FancyFeatureExtractor(PyFeatureExtractor):
        def extract_from(
            self,
            context: TuneContext,  # pylint: disable = unused-argument
            candidates: list[MeasureCandidate],  # pylint: disable = unused-argument
        ) -> list[np.ndarray]:
            return [tvm_tirx.runtime.tensor(np.random.rand(4, 5))]

    extractor = FancyFeatureExtractor()
    features = extractor.extract_from(TuneContext(), [])
    assert len(features) == 1
    assert features[0].shape == (4, 5)


if __name__ == "__main__":
    test_meta_schedule_feature_extractor()
