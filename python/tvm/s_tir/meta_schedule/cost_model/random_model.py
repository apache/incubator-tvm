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
Random cost model
"""

from tvm.s_tir.utils import derived_object

from ..cost_model import PyCostModel
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext


@derived_object
class RandomModel(PyCostModel):
    """Random cost model

    Parameters
    ----------
    random_state : Union[Tuple[str, np.ndarray, int, int, float], dict]
        The random state of the random number generator.
    path : Optional[str]
        The path of the random cost model.
    max_range : Optional[int]
        The maximum range of random results, [0, max_range].

    Reference
    ---------
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.get_state.html
    """

    import numpy as np  # type: ignore # pylint: disable=import-outside-toplevel

    random_state: tuple[str, np.ndarray, int, int, float] | dict
    path: str | None

    def __init__(
        self,
        *,
        seed: int | None = None,
        path: str | None = None,
        max_range: int | None = 100,
    ):
        import numpy as np  # type: ignore # pylint: disable=import-outside-toplevel

        super().__init__()
        if path is not None:
            self.load(path)
        else:
            np.random.seed(seed)
            self.random_state = np.random.get_state()
        self.max_range = max_range

    def load(self, path: str) -> None:
        """Load the cost model from given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        import numpy as np  # type: ignore # pylint: disable=import-outside-toplevel

        self.random_state = tuple(np.load(path, allow_pickle=True))  # type: ignore

    def save(self, path: str) -> None:
        """Save the cost model to given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        import numpy as np  # type: ignore # pylint: disable=import-outside-toplevel

        np.save(path, np.array(self.random_state, dtype=object), allow_pickle=True)

    def update(
        self,
        context: TuneContext,
        candidates: list[MeasureCandidate],
        results: list[RunnerResult],
    ) -> None:
        """Update the cost model given running results.

        Parameters
        ----------
        context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.
        results : List[RunnerResult]
            The running results of the measure candidates.
        """

    def predict(self, context: TuneContext, candidates: list[MeasureCandidate]) -> np.ndarray:  # type: ignore # pylint: disable=used-before-assignment
        """Update the cost model given running results.

        Parameters
        ----------
        context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.

        Return
        ------
        result : np.ndarray
            The predicted running results.
        """
        import numpy as np  # type: ignore # pylint: disable=import-outside-toplevel

        np.random.set_state(self.random_state)
        # TODO(@zxybazh): Use numpy's RandState object:
        # https://numpy.org/doc/1.16/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
        result = np.random.rand(len(candidates)) * self.max_range  # type: ignore
        self.random_state = np.random.get_state()
        return result
