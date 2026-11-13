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
"""XGBoost-based cost model"""

import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from itertools import chain as itertools_chain
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional

import numpy as np  # type: ignore

from tvm.s_tir.utils import derived_object
from tvm_tirx.runtime import Tensor
from tvm_tirx.support.tar import tar, untar

from ..cost_model import PyCostModel
from ..feature_extractor import FeatureExtractor
from ..logging import get_logger
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate
from ..utils import cpu_count, shash2hex
from .metric import max_curve

if TYPE_CHECKING:
    import xgboost as xgb  # type: ignore
    from xgboost.callback import TrainingCallback  # type: ignore

    from ..tune_context import TuneContext


logger = get_logger(__name__)  # pylint: disable=invalid-name


def make_metric_sorter(focused_metric):
    """Make sure the focused metric is the first one."""

    def metric_name_for_sort(name):
        if focused_metric == name:
            return "!" + name
        return name

    def sort_key(key):
        key, _ = key
        return metric_name_for_sort(key)

    return sort_key


class PackSum:
    """The pack-sum format

    Parameters
    ----------
    dmatrix : xgb.DMatrix
        A float64 array of shape [n, m],
        where `n` is the packed number of blocks,
        and `m` is the length of feature vector on each block
    ids : np.ndarray
        An int64 array of shape [n] containing nonnegative integers,
        indicating which the index of a sample that a block belongs to
    """

    dmatrix: "xgb.DMatrix"  # type: ignore # pylint: disable=invalid-name
    ids: np.ndarray

    def __init__(
        self,
        xs: list[np.ndarray],  # pylint: disable=invalid-name
        ys: np.ndarray | None,  # pylint: disable=invalid-name
    ):
        """Create PackSum format given a batch of samples

        Parameters
        ----------
        xs : List[np.ndarray]
            A batch of input samples
        ys : Optional[List[float]]
            A batch of labels. None means no labels available.
        """
        import xgboost as xgb  # type: ignore # pylint: disable=import-outside-toplevel

        repeats = [x.shape[0] for x in xs]
        xs = np.concatenate(xs, axis=0)
        self.ids = np.concatenate([[i] * repeat for i, repeat in enumerate(repeats)], axis=0)
        if ys is None:
            self.dmatrix = xgb.DMatrix(data=xs, label=None)
        else:
            ys = np.concatenate([[y] * repeat for y, repeat in zip(ys, repeats)], axis=0)
            self.dmatrix = xgb.DMatrix(data=xs, label=ys)
            self.dmatrix.set_weight(ys)

    def predict_with_score(self, pred: np.ndarray) -> np.ndarray:
        """Predict the labels given the block level prediction scores.

        Parameters
        ----------
        pred : np.ndarray
            The block level predictions

        Returns
        -------
        result : np.ndarray
            The predictions for each candidate.
        """
        return np.bincount(self.ids, weights=pred)

    def obj_square_error(self, ys_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Implement square error loss on pack-sum format as
        a custom objective function for xgboost.

        Parameters
        ----------
        ys_pred: np.ndarray
            The predictions

        Returns
        -------
        gradient: np.ndarray
            The gradient according to the xgboost format
        hessian: np.ndarray
            The hessian according to the xgboost format
        """
        # Making prediction
        ys_pred = self.predict_with_score(ys_pred)
        # Propagate prediction to each block
        ys_pred = ys_pred[self.ids]  # pylint: disable=invalid-sequence-index
        # The gradient and hessian
        ys = self.dmatrix.get_label()  # type: ignore # pylint: disable=invalid-name
        gradient = ys_pred - ys
        hessian = np.ones_like(gradient)
        return gradient * ys, hessian * ys

    def rmse(self, ys_pred: np.ndarray) -> tuple[str, float]:
        """Evaluate RMSE (rooted mean square error) in the pack-sum format

        Parameters
        ----------
        ys_pred: np.ndarray
            The raw predictions

        Returns
        -------
        name: str
            The name of the metric
        score: float
            The score of the metric
        """
        # Making prediction
        ys_pred = self.predict_with_score(ys_pred)
        # Propagate prediction to each block
        ys_pred = ys_pred[self.ids]  # pylint: disable=invalid-sequence-index
        # The RMSE
        ys = self.dmatrix.get_label()  # type: ignore # pylint: disable=invalid-name
        square_error = np.square(ys_pred - ys)
        rmse = np.sqrt(square_error.mean())
        return "p-rmse", rmse

    def average_peak_score(
        self,
        ys_pred: np.ndarray,
        n: int,
    ) -> tuple[str, float]:
        """Evaluate average-peak-score@N in the pack-sum format

        Parameters
        ----------
        ys_pred: np.ndarray
            The raw prediction
        n : int
            The N in average-peak-score@N

        Returns
        -------
        name: str
            The name of the metric
        score: float
            The score of the metric
        """
        ys = self.dmatrix.get_label()  # type: ignore # pylint: disable=invalid-name
        ys = self.predict_with_score(ys)  # type: ignore # pylint: disable=invalid-name
        ys = ys / np.unique(self.ids, return_counts=True)[1]  # type: ignore # pylint: disable=invalid-name
        ys_pred = self.predict_with_score(ys_pred)
        trials = np.argsort(ys_pred)[::-1][:n]
        trial_scores = ys[trials]
        curve = max_curve(trial_scores) / np.max(ys)
        score = np.mean(curve)
        return f"a-peak@{n}", score


class XGBConfig(NamedTuple):
    """XGBoost model configuration

    Reference: https://xgboost.readthedocs.io/en/stable/parameter.html

    Parameters
    ----------
    max_depth : int
        The maximum depth.
    gamma : float
        The gamma.
    min_child_weight : float
        The minimum child weight.
    eta : float
        The eta, learning rate.
    seed : int
        The random seed.
    nthread : Optional[int],
        The number of threads to use.
        Default is None, which means to use physical number of cores.
    tree_method : Literal["auto", "exact", "approx", "hist", "gpu_hist"]
        The tree construction algorithm used in XGBoost.
    """

    max_depth: int = 10
    gamma: float = 0.001
    min_child_weight: float = 0
    eta: float = 0.2
    seed: int = 43
    nthread: int | None = None
    tree_method: Literal["auto", "exact", "approx", "hist", "gpu_hist"] = "auto"

    def to_dict(self):
        """Convert to dict"""

        return {
            "max_depth": self.max_depth,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "eta": self.eta,
            "seed": self.seed,
            "nthread": self.nthread,
            "tree_method": self.tree_method,
        }


class FeatureGroup:
    """Feature group

    Parameters
    ----------
    group_hash : str
        The hash of the group
    features : List[np.ndarray]
        The features
    costs : List[float]
        The costs
    min_cost : float
        The minimum cost
    """

    group_hash: str
    features: list[np.ndarray]
    costs: np.ndarray
    min_cost: float

    def __init__(
        self,
        group_hash: str,
        features: list[np.ndarray],
        costs: np.ndarray,
    ) -> None:
        self.group_hash = group_hash
        self.features = features
        self.costs = costs
        self.min_cost = np.min(costs)

    def append(
        self,
        features: list[np.ndarray],
        costs: np.ndarray,
    ) -> None:
        self.features.extend(features)
        self.costs = np.append(self.costs, costs)
        self.min_cost = np.min(self.costs)


@derived_object
class XGBModel(PyCostModel):
    """XGBoost model

    Parameters
    ----------
    extractor : FeatureExtractor
        The feature extractor for the model.
    config : XGBConfig
        The XGBoost model config.
    num_warmup_samples : int
        The number of samples that are used for warmup, i.e., the first few samples are predicted
        with random results.
    early_stopping_rounds : int
        The number of rounds for early stopping.
    verbose_eval : int
        The verbose level when doing evaluation.
    average_peak_n : int
        The number to calculate average peak score.
    adaptive_training : bool
        Whether use adaptive training to reduce tuning time.
    """

    # feature extractor
    extractor: FeatureExtractor
    # xgboost model config
    config: XGBConfig
    # behavior of randomness
    num_warmup_samples: int
    # evaluation
    early_stopping_rounds: int
    verbose_eval: int
    average_peak_n: int
    # states
    data: dict[str, FeatureGroup]
    data_size: int
    booster: Optional["xgb.Booster"]
    # adaptive training
    adaptive_training: bool
    last_train_size: int

    def __init__(
        self,
        *,
        # feature extractor
        extractor: FeatureExtractor.FeatureExtractorType = "per-store-feature",
        # xgboost model config
        config: XGBConfig = XGBConfig(),
        # random result before enough samples
        num_warmup_samples: int = 100,
        # evaluation
        early_stopping_rounds: int = 50,
        verbose_eval: int = 25,
        average_peak_n: int = 32,
        adaptive_training: bool = True,
        num_tuning_cores: int | None = None,
        tree_method: Literal["auto", "exact", "approx", "hist", "gpu_hist"] | None = None,
    ):
        super().__init__()
        if not isinstance(extractor, FeatureExtractor):
            extractor = FeatureExtractor.create(extractor)
        # feature extractor
        self.extractor = extractor
        # model-related
        if config.nthread is None:
            # use physical core number
            if num_tuning_cores is None:
                config = config._replace(nthread=cpu_count(logical=False))
            else:
                config = config._replace(nthread=num_tuning_cores)

        if tree_method is not None:
            config._replace(tree_method=tree_method)

        self.config = config
        # behavior of randomness
        self.num_warmup_samples = num_warmup_samples
        # evaluation
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.average_peak_n = average_peak_n
        # states
        self.data = OrderedDict()
        self.data_size = 0
        self.booster = None
        # adaptive training
        self.adaptive_training = adaptive_training
        self.last_train_size = 0

    def load(self, path: str) -> None:
        """Load the cost model from given file location.

        Parameters
        ----------
        path : str
            The file path.

        Note
        ----
        Since XGBoost model trains from scratch, each time this method loads the model together with
        previously cached feature vectors and results, so that the subsequent training process could
        use all the existing data being stored on disk.
        """
        import xgboost as xgb  # pylint: disable=import-outside-toplevel

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.bin")
            data_path = os.path.join(tmp_dir, "data.npy")
            # Step 1. Untar
            untar(path, tmp_dir)
            # Step 2. Load data
            data = OrderedDict()
            data_size = 0
            for group_hash, features, costs in np.load(data_path, allow_pickle=True):
                data[group_hash] = FeatureGroup(
                    group_hash=group_hash,
                    features=list(features),
                    costs=costs,
                )
                data_size += len(costs)
            # Step 3. Load the model
            if os.path.exists(model_path):
                booster = xgb.Booster()
                booster.load_model(model_path)
            else:
                self.booster = None
        self.data = data
        self.data_size = data_size
        self.booster = booster

    def save(self, path: str) -> None:
        """Save the cost model to given file location.

        Parameters
        ----------
        path : str
            The file path.

        Note
        ----
        Since XGBoost model trains from scratch, each time this method saves the model together with
        previously cached feature vectors and results, so that the subsequent training process could
        use all the existing data being stored on disk.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.bin")
            data_path = os.path.join(tmp_dir, "data.npy")
            # Step 1. Save the model
            booster = self.booster
            if booster is not None:
                booster.save_model(model_path)
            else:
                model_path = None
            # Step 2. Save data
            data = [
                (
                    g.group_hash,
                    g.features,
                    g.costs,
                )
                for g in self.data.values()
            ]
            np.save(
                file=data_path,
                arr=np.array(data, dtype=object),
            )
            # Step 3. Tar it
            tar(path, [x for x in [model_path, data_path] if x is not None])
            logger.info("Saved XGBModel to %s", path)

    def update(
        self,
        context: "TuneContext",
        candidates: list[MeasureCandidate],
        results: list[RunnerResult],
    ) -> None:
        """Update the cost model given running results.

        Parameters
        ----------
        context : TuneContext
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.
        results : List[RunnerResult]
            The running results of the measure candidates.
        """
        assert len(candidates) == len(results)
        if len(candidates) == 0:
            return

        # Step 1. Get the feature group
        new_group_hash = shash2hex(context.mod)
        group = self.data.get(new_group_hash, None)

        # Step 2. Extract features
        def _feature(x: Tensor) -> np.ndarray:
            return x.numpy().astype("float32")

        def _mean_cost(x: RunnerResult) -> float:
            if not x.run_secs:
                return 1e10
            return float(np.median([float(s) for s in x.run_secs]))

        new_features = [_feature(x) for x in self.extractor.extract_from(context, candidates)]
        new_mean_costs = [_mean_cost(x) for x in results]

        # Filter instances with no features
        new_mean_costs = [c for i, c in enumerate(new_mean_costs) if len(new_features[i]) != 0]
        new_mean_costs_np = np.array(new_mean_costs).astype("float32")
        new_features = [f for f in new_features if len(f) != 0]
        if not new_features:
            return

        # Steps 3. Run validation
        if group is not None and self.booster is not None:
            logger.debug(
                "XGB validation: %s",
                "\t".join(
                    f"{key}: {score:.6f}"
                    for key, score in self._validate(
                        xs=new_features,
                        ys=group.min_cost / new_mean_costs_np,
                    )
                ),
            )

        # Step 4. Add the features into the data points
        if group is None:
            group = FeatureGroup(
                group_hash=new_group_hash,
                features=new_features,
                costs=new_mean_costs_np,
            )
        else:
            group.append(new_features, new_mean_costs_np)
        self.data[new_group_hash] = group
        self.data_size += len(new_features)

        if (
            self.adaptive_training
            and self.data_size - self.last_train_size < self.last_train_size / 5
        ):
            # Set a training threshold related to `last_train_size` to reduce the training
            # overhead when there're too many results
            return
        self.last_train_size = self.data_size

        # Step 5. Re-train the model
        with np.errstate(divide="ignore", invalid="ignore"):
            feature_list = list(
                itertools_chain.from_iterable([g.features for g in self.data.values()])
            )
            cost_ratio_list = [
                np.divide(g.min_cost, g.costs, out=np.zeros_like(g.costs), where=g.costs != 0)
                for g in self.data.values()
            ]
            cost_ratios = np.concatenate(cost_ratio_list, axis=0)

        self._train(xs=feature_list, ys=cost_ratios)

    def predict(
        self,
        context: "TuneContext",
        candidates: list[MeasureCandidate],
    ) -> np.ndarray:
        """Predict the normalized score using the cost model.

        Parameters
        ----------
        context : TuneContext
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.

        Return
        ------
        result : np.ndarray
            The predicted normalized score.
        """
        if self.data_size >= self.num_warmup_samples and self.booster is not None:
            ret = self._predict(
                xs=[
                    x.numpy().astype("float32")
                    for x in self.extractor.extract_from(
                        context,
                        candidates,
                    )
                ]
            )
        else:
            ret = np.random.uniform(
                low=0,
                high=1,
                size=(len(candidates),),
            )
        return ret.astype("float64")

    def _train(  # type: ignore # pylint: disable=invalid-name
        self,
        xs: list[np.ndarray],
        ys: np.ndarray,
    ) -> None:
        import xgboost as xgb  # type: ignore # pylint: disable=import-outside-toplevel

        self.d_train = PackSum(xs=xs, ys=ys)

        def obj(ys_pred: np.ndarray, d_train: "xgb.DMatrix"):  # type: ignore # pylint: disable = unused-argument
            return self.d_train.obj_square_error(ys_pred)

        def rmse(ys_pred: np.ndarray, d_train: "xgb.DMatrix"):  # type: ignore # pylint: disable = unused-argument
            return self.d_train.rmse(ys_pred)

        def avg_peak_score(ys_pred: np.ndarray, d_train: "xgb.DMatrix"):  # type: ignore # pylint: disable = unused-argument
            return self.d_train.average_peak_score(ys_pred, self.average_peak_n)

        self.booster = xgb.train(
            self.config.to_dict(),
            self.d_train.dmatrix,
            num_boost_round=10000,
            obj=obj,
            callbacks=[
                _get_custom_call_back(
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=self.verbose_eval,
                    fevals=[rmse, avg_peak_score],
                    evals=[(self.d_train.dmatrix, "tr")],
                    cvfolds=None,
                )
            ],
        )

        del self.d_train

    def _predict(  # type: ignore # pylint: disable=invalid-name
        self,
        xs: list[np.ndarray],
    ) -> np.ndarray:
        d_test = PackSum(xs=xs, ys=None)
        pred = self.booster.predict(d_test.dmatrix)
        ret = d_test.predict_with_score(pred)
        return ret

    def _validate(  # type: ignore # pylint: disable=invalid-name
        self,
        xs: list[np.ndarray],
        ys: np.ndarray,
    ) -> list[tuple[str, float]]:
        """Evaluate the score of inputs.

        Parameters
        ----------
        xs : List[np.ndarray]
            A batch of input samples
        ys : List[float]
            A batch of labels

        Returns
        -------
        scores: np.ndarray
            The predicted result for all inputs.
        """
        assert self.booster is not None

        d_valid = PackSum(xs=xs, ys=ys)

        def average_peak_score(ys_pred: np.ndarray):
            return d_valid.average_peak_score(ys_pred, n=self.average_peak_n)

        ys_pred = self.booster.predict(d_valid.dmatrix)
        eval_result: list[tuple[str, float]] = [
            feval(ys_pred)
            for feval in (
                average_peak_score,
                d_valid.rmse,
            )
        ]
        eval_result.sort(key=make_metric_sorter("p-rmse"))
        return eval_result


def _get_custom_call_back(
    early_stopping_rounds: int,
    verbose_eval: int,
    fevals: list[Callable],
    evals: list[tuple["xgb.DMatrix", str]],
    focused_metric: str = "tr-p-rmse",
    cvfolds: list["xgb.training.CVPack"] | None = None,
) -> "TrainingCallback":
    """Get a customized callback function for XGBoost. Work around xgboost import."""

    def optional_xgboost_callback(cls):
        """Decorator for importing TrainingCallback from xgboost"""
        # pylint:disable = import-outside-toplevel
        try:
            from xgboost.callback import TrainingCallback  # type: ignore
        # pylint:enable = import-outside-toplevel
        except ImportError:

            class TrainingCallback:  # type: ignore
                pass

        class OptXGBoostCustomCallback(cls, TrainingCallback):  # type: ignore
            pass

        return OptXGBoostCustomCallback

    @optional_xgboost_callback
    class XGBoostCustomCallback:
        """Custom callback class for xgboost to support multiple custom evaluation functions"""

        def __init__(
            self,
            early_stopping_rounds: int,
            verbose_eval: int,
            fevals: list[Callable],
            evals: list[tuple["xgb.DMatrix", str]],
            focused_metric: str = "tr-p-rmse",
            cvfolds: list["xgb.training.CVPack"] | None = None,
        ):
            self.early_stopping_rounds = early_stopping_rounds
            self.verbose_eval = verbose_eval
            self.fevals = fevals
            self.evals = evals
            self.state: dict[str, Any] = {}
            self.focused_metric = focused_metric
            self.sort_key = make_metric_sorter(focused_metric=focused_metric)
            self.cvfolds = cvfolds
            if cvfolds is not None:
                self.aggregated_cv = None

        def __call__(self, env: "xgb.core.CallbackEnv"):
            # Compatibility with xgboost < 1.3
            return self.after_iteration(env.model, env.iteration, env.evaluation_result_list)

        def init(self, model: "xgb.Booster"):
            """Internal function for initialization"""
            booster: xgb.Booster = model
            self.state["best_iteration"] = 0
            self.state["best_score"] = float("inf")
            if booster is None:
                assert self.cvfolds is not None
                return
            if booster.attr("best_score") is not None:
                self.state["best_score"] = float(booster.attr("best_score"))
                self.state["best_iteration"] = int(booster.attr("best_iteration"))
                self.state["best_msg"] = booster.attr("best_msg")
            else:
                booster.set_attr(best_iteration=str(self.state["best_iteration"]))
                booster.set_attr(best_score=str(self.state["best_score"]))

        def after_iteration(self, model: "xgb.Booster", epoch: int, evals_log: dict):  # pylint: disable = unused-argument
            """Internal function for after_iteration"""
            # pylint:disable = import-outside-toplevel
            try:
                from xgboost.callback import _fmt_metric  # type: ignore
            except ImportError:
                # Compatibility with xgboost >= 1.6

                def _fmt_metric(value, show_stdv=True):
                    if len(value) == 2:
                        return f"{value[0]}:{value[1]:.5f}"
                    if len(value) == 3:
                        if show_stdv:
                            return f"{value[0]}:{value[1]:.5f}+{value[2]:.5f}"
                        return f"{value[0]}:{value[1]:.5f}"
                    raise ValueError("wrong metric value", value)

            import xgboost as xgb

            # make it compatible with xgboost<1.7
            try:
                from xgboost import rabit as collective  # type: ignore
            except ImportError:
                from xgboost import collective  # type: ignore

            try:
                from xgboost.training import aggcv  # type: ignore
            except ImportError:
                from xgboost.callback import _aggcv as aggcv  # type: ignore

            # pylint:enable = import-outside-toplevel
            if not self.state:
                self.init(model)
            booster: xgb.Booster = model
            iteration: int = epoch
            cvfolds: list[xgb.training.CVPack] = self.cvfolds
            ##### Evaluation #####
            # `eval_result` is a list of (key, score)
            eval_result: list[tuple[str, float]] = []
            if cvfolds is None:
                eval_result = list(
                    itertools_chain.from_iterable(
                        [
                            (key, float(value))
                            for key, value in map(
                                lambda x: x.split(":"),
                                booster.eval_set(
                                    evals=self.evals,
                                    iteration=iteration,
                                    feval=feval,
                                ).split()[1:],
                            )
                        ]
                        for feval in self.fevals
                    )
                )
            else:
                eval_result = list(
                    itertools_chain.from_iterable(
                        [
                            (key, score)
                            for key, score, _std in aggcv(
                                fold.eval(
                                    iteration=iteration,
                                    feval=feval,
                                )
                                for fold in cvfolds
                            )
                        ]
                        for feval in self.fevals
                    )
                )
            eval_result = list(eval_result)
            eval_result.sort(key=self.sort_key)

            ##### Print eval result #####
            if self.verbose_eval and iteration % self.verbose_eval == 0:
                info = []
                for key, score in eval_result:
                    if "null" not in key:
                        info.append(f"{key}: {score:.6f}")
                logger.debug("XGB iter %3d: %s", iteration, "\t".join(info))

            ##### Choose score and do early stopping #####
            score = None
            for key, _score in eval_result:
                if key == self.focused_metric:
                    score = _score
                    break
            assert score is not None

            best_score = self.state["best_score"]
            best_iteration = self.state["best_iteration"]
            if score < best_score:
                tab = "\t"  # to work with f-string
                msg = f"[{epoch}] {tab.join([_fmt_metric(x) for x in eval_result])}"
                self.state["best_msg"] = msg
                self.state["best_score"] = score
                self.state["best_iteration"] = epoch
                # save the property to attributes, so they will occur in checkpoint.
                if model is not None:
                    model.set_attr(
                        best_score=str(self.state["best_score"]),
                        best_iteration=str(self.state["best_iteration"]),
                        best_msg=self.state["best_msg"],
                    )
            elif epoch - best_iteration >= self.early_stopping_rounds:
                best_msg = self.state["best_msg"]

                if self.verbose_eval and collective.get_rank() == 0:
                    logger.debug("XGB stopped. Best iteration: %s ", best_msg)
                # instead of raising EarlyStopException, returning True to end the training
                return True
            # False to indicate training should not stop.
            return False

    return XGBoostCustomCallback(
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        fevals=fevals,
        evals=evals,
        focused_metric=focused_metric,
        cvfolds=cvfolds,
    )
