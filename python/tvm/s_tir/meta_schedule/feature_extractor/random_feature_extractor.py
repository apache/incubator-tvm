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
"""Random Feature Extractor."""

import numpy as np  # type: ignore

import tvm_tirx.runtime
from tvm.s_tir.utils import derived_object

from ..feature_extractor import PyFeatureExtractor
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext


@derived_object
class RandomFeatureExtractor(PyFeatureExtractor):
    """Random Feature Extractor

    Parameters
    ----------
    feature_size : int
        The size of each block's feature vector.
    max_block_num : int
        The maximum number of blocks in each schedule.
    random_state : Union[Tuple[str, np.ndarray, int, int, float], dict]
        The current random state of the f
    """

    feature_size: int
    max_block_num: int
    random_state: tuple[str, np.ndarray, int, int, float] | dict

    def __init__(self, *, feature_size: int = 30, max_block_num: int = 5, seed=0):
        super().__init__()
        assert max_block_num >= 1, "Max block number must be greater or equal to one!"
        self.max_block_num = max_block_num
        self.feature_size = feature_size
        np.random.seed(seed)
        self.random_state = np.random.get_state()

    def extract_from(
        self, context: TuneContext, candidates: list[MeasureCandidate]
    ) -> list[tvm_tirx.runtime.Tensor]:
        np.random.set_state(self.random_state)
        result = [
            np.random.rand(np.random.randint(1, self.max_block_num + 1), self.feature_size)
            for candidate in candidates
        ]
        self.random_state = np.random.get_state()
        return [tvm_tirx.runtime.tensor(x) for x in result]
