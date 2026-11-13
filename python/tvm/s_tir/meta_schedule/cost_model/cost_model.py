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
# ruff: noqa: RUF012
"""Meta Schedule CostModel."""

import ctypes
from collections.abc import Callable
from typing import Union

import tvm_ffi

# isort: off
from typing import Literal

# isort: on

import numpy as np  # type: ignore
from tvm_ffi import register_object

from .. import _ffi_api
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext


@register_object("s_tir.meta_schedule.CostModel")
class CostModel(tvm_ffi.Object):
    """Cost model."""

    CostModelType = Union["CostModel", Literal["xgb", "mlp", "random"]]

    def load(self, path: str) -> None:
        """Load the cost model from given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        _ffi_api.CostModelLoad(self, path)  # type: ignore # pylint: disable=no-member

    def save(self, path: str) -> None:
        """Save the cost model to given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        _ffi_api.CostModelSave(self, path)  # type: ignore # pylint: disable=no-member

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
        _ffi_api.CostModelUpdate(self, context, candidates, results)  # type: ignore # pylint: disable=no-member

    def predict(self, context: TuneContext, candidates: list[MeasureCandidate]) -> np.ndarray:
        """Predict normalized score with the cost model.

        Parameters
        ----------
        context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.

        Return
        ------
        result : np.ndarray
            The predicted normalized score.
        """
        n = len(candidates)
        results = np.zeros(shape=(n,), dtype="float64")
        _ffi_api.CostModelPredict(  # type: ignore # pylint: disable=no-member
            self,
            context,
            candidates,
            results.ctypes.data_as(ctypes.c_void_p),
        )
        return results

    @staticmethod
    def create(
        kind: Literal["xgb", "mlp", "random", "none"],
        *args,
        **kwargs,
    ) -> "CostModel":
        """Create a CostModel.

        Parameters
        ----------
        kind : Literal["xgb", "mlp", "random", "none"]
            The kind of the cost model. Can be "xgb", "mlp", "random" or "none".

        Returns
        -------
        cost_model : CostModel
            The created cost model.
        """
        from . import RandomModel, XGBModel  # pylint: disable=import-outside-toplevel

        if kind == "xgb":
            return XGBModel(*args, **kwargs)  # type: ignore

        # params only relevant to XGBModel
        _xgb_params = ["num_tuning_cores", "tree_method"]

        for param in _xgb_params:
            if param in kwargs:
                kwargs.pop(param)

        if kind == "random":
            return RandomModel(*args, **kwargs)  # type: ignore
        if kind == "mlp":
            from .mlp_model import (  # type: ignore  # pylint: disable=import-outside-toplevel
                MLPModel,
            )

            return MLPModel(*args, **kwargs)  # type: ignore
        if kind == "none":
            return None  # no cost model required
        raise ValueError(f"Unknown CostModel: {kind}")


create = CostModel.create  # pylint: disable=invalid-name


@register_object("s_tir.meta_schedule.PyCostModel")
class _PyCostModel(CostModel):
    """
    A TVM object cost model to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyCostModel
    """

    def __init__(
        self,
        f_load: Callable | None = None,
        f_save: Callable | None = None,
        f_update: Callable | None = None,
        predict_func: Callable | None = None,
    ):
        """Constructor."""

        def f_predict(context: TuneContext, candidates: list[MeasureCandidate], return_ptr) -> None:
            n = len(candidates)
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_double))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(n,))
            res = predict_func(context, candidates)
            array_wrapper[:] = res
            assert array_wrapper.dtype == "float64", (
                "ValueError: Invalid data type returned from CostModel Predict!"
            )

        self.__init_handle_by_constructor__(
            _ffi_api.CostModelPyCostModel,  # type: ignore # pylint: disable=no-member
            f_load,
            f_save,
            f_update,
            f_predict,
        )


class PyCostModel:
    """
    An abstract cost model with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyCostModel,
        "methods": ["load", "save", "update", "predict"],
    }

    def load(self, path: str) -> None:
        """Load the cost model from given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the cost model to given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def predict(self, context: TuneContext, candidates: list[MeasureCandidate]) -> np.ndarray:
        """Predict given the measure candidates.

        Parameters
        ----------
        context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.

        Return
        ------
        result : np.ndarray
            The predicted normalized score.
        """
        raise NotImplementedError
