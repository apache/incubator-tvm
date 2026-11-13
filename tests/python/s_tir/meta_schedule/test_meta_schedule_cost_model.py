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
# pylint: disable=missing-docstring
# ruff: noqa: F401
import os
import shutil
import tempfile
import unittest
from functools import partial
from importlib.util import find_spec

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.cost_model import PyCostModel, RandomModel, XGBModel
from tvm.s_tir.meta_schedule.cost_model.xgb_model import PackSum, _get_custom_call_back
from tvm.s_tir.meta_schedule.feature_extractor import RandomFeatureExtractor
from tvm.s_tir.meta_schedule.runner import RunnerResult
from tvm.s_tir.meta_schedule.search_strategy import MeasureCandidate
from tvm.s_tir.meta_schedule.tune_context import TuneContext
from tvm.s_tir.schedule.schedule import Schedule
from tvm.s_tir.utils import derived_object
from tvm_tirx.script import tirx as T

requires_xgboost = pytest.mark.skipif(
    find_spec("xgboost") is None, reason="xgboost is not installed"
)


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring
@tvm_tirx.script.ir_module
class Matmul:
    @Ts.function
    def main(
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        C: T.Tensor((1024, 1024), "float32"),
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm_tirx.script.ir_module
class FullModule:
    @Ts.function
    def main(T_full: T.Tensor((T.int64(2), T.int64(3)), "float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
            with Ts.sblock("T_full"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads()
                Ts.writes(T_full[v_ax0, v_ax1])
                T_full[v_ax0, v_ax1] = T.float32(1)


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,disable=unused-argument


def test_meta_schedule_cost_model():
    @derived_object
    class FancyCostModel(PyCostModel):
        def load(self, path: str) -> None:
            pass

        def save(self, path: str) -> None:
            pass

        def update(
            self,
            context: TuneContext,
            candidates: list[MeasureCandidate],
            results: list[RunnerResult],
        ) -> None:
            pass

        def predict(self, context: TuneContext, candidates: list[MeasureCandidate]) -> np.ndarray:
            return np.random.rand(10)

    model = FancyCostModel()
    model.save("fancy_test_location")
    model.load("fancy_test_location")
    model.update(TuneContext(), [], [])
    results = model.predict(
        TuneContext(), [MeasureCandidate(Schedule(mod=Matmul), []) for _ in range(10)]
    )
    assert results.shape == (10,)


def test_meta_schedule_random_model():
    model = RandomModel()
    model.update(TuneContext(), [], [])
    res = model.predict(TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(10)])
    assert len(res) == 10
    assert min(res) >= 0 and max(res) <= model.max_range


def test_meta_schedule_random_model_reseed():
    model = RandomModel(seed=100)
    res = model.predict(TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(20)])
    new_model = RandomModel(seed=100)
    new_res = new_model.predict(
        TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(20)]
    )
    assert (res == new_res).all()


def test_meta_schedule_random_model_reload():
    model = RandomModel(seed=25973)
    model.predict(
        TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(30)]
    )  # change state
    path = os.path.join(tempfile.mkdtemp(), "test_output_meta_schedule_random_model.npy")
    model.save(path)
    res1 = model.predict(TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(70)])
    model.load(path)
    res2 = model.predict(TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(70)])
    shutil.rmtree(os.path.dirname(path))
    assert (res1 == res2).all()


def _dummy_candidate():
    return MeasureCandidate(Schedule(Matmul), [])


def _dummy_result(num_samples: int = 4, max_run_sec: int = 10):
    return RunnerResult(list(np.random.rand(num_samples) * max_run_sec + 1e-6), None)


@requires_xgboost
def test_meta_schedule_xgb_model():
    extractor = RandomFeatureExtractor()
    model = XGBModel(extractor=extractor, num_warmup_samples=2, num_tuning_cores=1)
    update_sample_count = 10
    predict_sample_count = 100
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.predict(TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)])


@requires_xgboost
def test_meta_schedule_xgb_model_no_feature():
    model = XGBModel(num_warmup_samples=0, num_tuning_cores=1)
    tune_ctx = TuneContext(
        FullModule,
        num_threads=1,
        target={"kind": "llvm", "num-cores": 16},
        space_generator="post-order-apply",
        search_strategy="evolutionary",
    )
    candidate = MeasureCandidate(Schedule(FullModule), [])
    model.update(tune_ctx, [candidate], [_dummy_result()])
    model.predict(tune_ctx, [candidate])


@requires_xgboost
def test_meta_schedule_xgb_model_reload():
    extractor = RandomFeatureExtractor()
    model = XGBModel(extractor=extractor, num_warmup_samples=10, num_tuning_cores=1)
    update_sample_count = 20
    predict_sample_count = 30
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.predict(TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)])
    with tempfile.NamedTemporaryFile() as path:
        # Backup
        random_state = model.extractor.random_state  # save feature extractor's random state
        old_data = model.data
        old_data_size = model.data_size
        model.save(path.name)
        res1 = model.predict(
            TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)]
        )
        # Load
        model.extractor.random_state = random_state  # load feature extractor's random state
        model.load(path.name)
        new_data = model.data
        new_data_size = model.data_size
        res2 = model.predict(
            TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)]
        )
    assert (res1 == res2).all()
    assert old_data_size == new_data_size
    assert len(old_data) == len(new_data)
    for (k1, g1), (k2, g2) in zip(  # pylint: disable=invalid-name
        old_data.items(), new_data.items()
    ):
        assert k1 == k2
        assert k1 == g1.group_hash
        assert k2 == g2.group_hash
        assert (g1.costs == g2.costs).all()
        assert len(g1.features) == len(g2.features)
        for f1, f2 in zip(g1.features, g2.features):  # pylint: disable=invalid-name
            assert (f1 == f2).all()


@requires_xgboost
def test_meta_schedule_xgb_model_reupdate():
    extractor = RandomFeatureExtractor()
    model = XGBModel(extractor=extractor, num_warmup_samples=2, num_tuning_cores=1)
    update_sample_count = 60
    predict_sample_count = 100
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.predict(TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)])


@requires_xgboost
def test_meta_schedule_xgb_model_callback_as_function():
    # pylint: disable=import-outside-toplevel
    from itertools import chain as itertools_chain

    import xgboost as xgb

    # pylint: enable=import-outside-toplevel

    extractor = RandomFeatureExtractor()
    model = XGBModel(extractor=extractor, num_warmup_samples=10, num_tuning_cores=1)
    update_sample_count = 20
    predict_sample_count = 30

    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.predict(TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)])
    with tempfile.NamedTemporaryFile() as path:
        # Backup and train on new TrainingCallBack api
        random_state = model.extractor.random_state  # save feature extractor's random state

        model.save(path.name)

        old_booster = model.booster
        xs = [  # pylint: disable=invalid-name
            x.numpy().astype("float32")
            for x in extractor.extract_from(
                TuneContext(),
                [_dummy_candidate() for i in range(predict_sample_count)],
            )
        ]
        d_test = PackSum(xs=xs, ys=None)
        pred1 = old_booster.predict(d_test.dmatrix)

        # Load and train on deprecated TrainingCallBack api
        model.extractor.random_state = random_state  # load feature extractor's random state
        model.load(path.name)
        d_train = PackSum(
            xs=list(itertools_chain.from_iterable([g.features for g in model.data.values()])),
            ys=np.concatenate(
                [g.min_cost / g.costs for g in model.data.values()],
                axis=0,
            ),
        )

        def obj(ys_pred: np.ndarray, d_train1: "xgb.DMatrix"):  # type: ignore # pylint: disable = unused-argument
            return d_train.obj_square_error(ys_pred)

        def rmse(ys_pred: np.ndarray, d_train1: "xgb.DMatrix"):  # type: ignore # pylint: disable = unused-argument
            return d_train.rmse(ys_pred)

        def avg_peak_score(ys_pred: np.ndarray, d_train1: "xgb.DMatrix"):  # type: ignore # pylint: disable = unused-argument
            return d_train.average_peak_score(ys_pred, model.average_peak_n)

        new_booster = xgb.train(
            model.config.to_dict(),
            d_train.dmatrix,
            num_boost_round=10000,
            obj=obj,
            callbacks=[
                _get_custom_call_back(
                    early_stopping_rounds=model.early_stopping_rounds,
                    verbose_eval=model.verbose_eval,
                    fevals=[rmse, avg_peak_score],
                    evals=[(d_train.dmatrix, "tr")],
                    cvfolds=None,
                )
            ],
        )

        xs = [  # pylint: disable=invalid-name
            x.numpy().astype("float32")
            for x in extractor.extract_from(
                TuneContext(),
                [_dummy_candidate() for i in range(predict_sample_count)],
            )
        ]
        d_test = PackSum(xs=xs, ys=None)
        pred2 = new_booster.predict(d_test.dmatrix)

    assert np.allclose(pred1, pred2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tvm_tirx.testing.main()
