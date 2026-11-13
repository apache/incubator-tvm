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
# type: ignore[import]
"""
Segment Sum MLP cost model
"""

import glob
import math
import os
import random
import tempfile
from collections import OrderedDict
from itertools import chain as itertools_chain
from typing import NamedTuple

import numpy as np  # type: ignore
import torch  # type: ignore

from tvm.s_tir.utils import derived_object
from tvm_tirx.runtime import Tensor
from tvm_tirx.support.tar import tar, untar
from tvm_tirx.target import Target

from ..cost_model import PyCostModel
from ..database import JSONDatabase
from ..feature_extractor import FeatureExtractor, PerStoreFeature
from ..logging import get_logger
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext
from ..utils import shash2hex

logger = get_logger("mlp_model")  # pylint: disable=invalid-name

# pylint: disable=no-member,import-outside-toplevel


class SegmentSumMLPConfig(NamedTuple):
    """SegmentSum MLP model configuration

    Parameters
    ----------
    input_dim : int
        The input dim for the model.
    hidden_dim : int
        The hidden dim for the model.
    output_dim : int
        The output dim for the model.
    use_norm : bool
        Whether to normalize the segment sum or not.
    use_sigmoid : bool
        Whether to use sigmoid on the final output or not.
    """

    input_dim: int = 172
    hidden_dim: int = 256
    output_dim: int = 1
    use_norm: bool = False
    use_sigmoid: bool = False

    def to_dict(self):  # pylint: disable=missing-function-docstring
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "use_norm": self.use_norm,
            "use_sigmoid": self.use_sigmoid,
        }


class TrainerConfig(NamedTuple):
    """Trainer configuration

    Parameters
    ----------
    batch_size : int
        The batch size.
    learning rate : float
        The learning rate.
    weight decay : float
        The weight decay.
    num_epoch_full : int
        The number of epochs used in full training.
    num_epoch_incremental : int
        The number of epochs used in incremental training.
    grad_clip_norm: float
        The norm of gradient clipping.
    train_verbose: int
        The verbose frequency for training in batches.
    test_interval: int
        The testing interval in epochs.
    test_split: float
        The fraction of data for testing.
    frozen: bool
        Determine whether to re-train the model or not.
    """

    batch_size: int = 128
    learning_rate: float = 7e-4
    weight_decay: float = 1e-6
    num_epoch_full: int = 50
    num_epoch_incremental: int = 5
    grad_clip_norm: float = 0.5
    train_verbose: int = 1000
    test_interval: int = 1
    test_split: float = 0.2
    frozen: bool = False

    def to_dict(self):  # pylint: disable=missing-function-docstring
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epoch_full": self.num_epoch_full,
            "num_epoch_incremental": self.num_epoch_incremental,
            "grad_clip_norm": self.grad_clip_norm,
            "train_verbose": self.train_verbose,
            "test_interval": self.test_interval,
            "test_split": self.test_split,
            "frozen": self.frozen,
        }


# pylint: disable=too-few-public-methods
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

    def append(  # pylint: disable=missing-function-docstring
        self,
        features: list[np.ndarray],
        costs: np.ndarray,
    ) -> None:
        self.features.extend(features)
        self.costs = np.append(self.costs, costs)
        self.min_cost = np.min(self.costs)


# pylint: disable=too-many-instance-attributes
class SegmentDataLoader:
    """Dataloader for Segment Sum MLP model.

    Parameters
    ----------
    features : List[np.ndarray]
        The features
    results : np.ndarray
        The measured results, can be None.
    batch_size : int
        The batch size
    shuffle : bool
        Whether to shuffle the dataset or not
    """

    def __init__(
        self,
        features,
        results=None,
        batch_size=128,
        shuffle=True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(features)

        # flatten features and store the starting indices
        self.segment_sizes = torch.tensor([len(feature) for feature in features], dtype=torch.int32)
        self.feature_offsets = (
            torch.cumsum(self.segment_sizes, 0, dtype=torch.int32) - self.segment_sizes
        )
        features = torch.cat([torch.tensor(feature) for feature in features])
        norm, _ = features.max(dim=0)
        norm[norm == 0] = 1
        self.features = features / norm
        self.results = torch.tensor(results) if results is not None else None
        self.iter_order = self.pointer = None

    def __len__(self):
        return self.data_size

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.data_size)
        else:
            self.iter_order = torch.arange(self.data_size)
        self.pointer = 0
        return self

    def __next__(self):
        if self.pointer >= self.data_size:
            raise StopIteration
        batch_indices = self.iter_order[self.pointer : self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):
        segment_sizes, feature_offsets = self.segment_sizes[indices], self.feature_offsets[indices]
        feature_indices = torch.empty(segment_sizes.sum(), dtype=torch.int32)
        idx = 0
        for offset, seg_size in zip(feature_offsets, segment_sizes):
            feature_indices[idx : idx + seg_size] = torch.arange(offset, offset + seg_size)
            idx += seg_size
        features = self.features[feature_indices.long()]
        results = None
        if self.results is not None:
            results = self.results[indices.long()]
        return segment_sizes, features, results


def lambda_rank_loss(  # pylint: disable=too-many-locals
    preds: "torch.Tensor",
    labels: "torch.Tensor",
    k: int | None = None,
    eps: float = 1e-10,
    sigma: float = 1.0,
) -> "torch.Tensor":
    """
    LambdaLoss: Metric-Driven Loss for Learning-to-Rank

    Parameters
    ----------
    preds : Tensor
        The predicted runtime for each candidate.
    labels : Tensor
        The measured runtime for each candidate.
    k : int
        Loss for top k.
        Default is None, which means computing all scores.
    eps : float
        The minimum value to the denominator and argument of log if they reach 0.
    sigma : float
        The scaling factor to the input of the sigmoid function.

    Returns
    -------
    loss : Tensor
        The lambda rank loss.
    """
    device = preds.device
    y_pred, y_true = preds[None, :], labels[None, :]
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs) & (true_diffs > 0)
    ndcg_at_k_mask = torch.zeros(
        (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device
    )
    ndcg_at_k_mask[:k, :k] = 1
    true_sorted_by_preds.clamp_(min=0.0)
    y_true_sorted.clamp_(min=0.0)
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1.0 + pos_idxs.float())[None, :]  # pylint: disable=invalid-name
    maxDCGs = torch.sum(  # pylint: disable=invalid-name
        ((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1
    ).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]  # pylint: disable=invalid-name
    weights = torch.abs(
        torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
    ) * torch.abs(G[:, :, None] - G[:, None, :])
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs[torch.isnan(scores_diffs)] = 0.0
    weighted_probs = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    losses = torch.log2(weighted_probs)
    masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
    loss = -torch.sum(masked_losses)
    return loss


def topk_score(
    pred_results: "torch.Tensor",
    gt_results: "torch.Tensor",
    k: int,
) -> float:
    """
    Evaluate the top-k score

    Parameters
    ----------
    pred_results: Tensor
        The raw prediction
    gt_results: Tensor
        The measured results
    k : int
        The k in top k score

    Returns
    -------
    score : float
        The top-k score
    """
    k = min(k, len(pred_results))
    topk_indices = torch.topk(pred_results, k, largest=False).indices
    score = gt_results.min() / gt_results[topk_indices].min()
    return score.item()


class SegmentSumMLP(torch.nn.Module):
    """Segment Sum MLP model.

    Parameters
    ----------
    input_dim : int
        The input dim for the model.
    hidden_dim : int
        The hidden dim for the model.
    output_dim : int
        The output dim for the model.
    use_norm : bool
        Whether to normalize the segment sum or not.
    use_sigmoid : bool
        Whether to use sigmoid on the final output or not.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    use_norm: bool
    use_sigmoid: bool

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_dim: int = 172,
        hidden_dim: int = 256,
        output_dim: int = 1,
        use_norm: bool = False,
        use_sigmoid: bool = False,
    ):
        from torch import nn  # type: ignore

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.norm = nn.BatchNorm1d(hidden_dim) if use_norm else nn.Identity()
        self.layer0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(  # pylint: disable=missing-function-docstring
        self,
        segment_sizes: "torch.Tensor",
        features: "torch.Tensor",
    ) -> "torch.Tensor":
        n_seg = len(segment_sizes)
        encoded_features = self.encoder(features)
        segment_indices = torch.repeat_interleave(
            torch.arange(n_seg, device=features.device),
            segment_sizes.long(),
        )
        n_dim = encoded_features.shape[1]
        segment_sum = torch.scatter_add(
            input=torch.zeros((n_seg, n_dim), dtype=encoded_features.dtype, device=features.device),
            dim=0,
            index=segment_indices.view(-1, 1).expand(-1, n_dim),
            src=encoded_features,
        )
        out = self.norm(segment_sum)
        out = self.layer0(out) + out
        out = self.layer1(out) + out
        out = self.decoder(out).squeeze()
        out = self.sigmoid(out)
        return out


def extract_features(
    context: TuneContext,
    candidates: list[MeasureCandidate],
    results: list[RunnerResult] | None = None,
    extractor: FeatureExtractor | None = None,
):
    """Extract feature vectors and compute mean costs.

    Parameters
    ----------
    context: TuneContext
        The tuning context.
    candidates: List[MeasureCandidate]
        The measure candidates.
    results: Optional[List[RunnerResult]]
        The measured results, can be None if used in prediction.
    extractor: Optional[FeatureExtractor]
        The feature extractor.

    Returns
    -------
    new_features: List[np.ndarray]
        The extracted features.
    new_mean_costs: np.ndarray
        The mean costs.
    """
    extractor = extractor or PerStoreFeature(extract_workload=True)

    def _feature(feature: Tensor) -> np.ndarray:
        return feature.numpy().astype("float32")

    def _mean_cost(res: RunnerResult) -> float:
        if not res.run_secs:
            return 1e10
        return float(np.median([float(s) for s in res.run_secs]))

    new_features = [_feature(x) for x in extractor.extract_from(context, candidates)]
    new_mean_costs = (
        np.array([_mean_cost(x) for x in results]).astype("float32")
        if results is not None
        else None
    )
    return new_features, new_mean_costs


class State:
    """State of the trainer

    Parameters
    ----------
    model: SegmentSumMLP
        The cost model.
    data: Dict[str, FeatureGroup]
        The data groups.
    data_size: int
        The size of all data.
    untrained_size: int
        The size of the untrained data.
    """

    model: SegmentSumMLP
    data: dict[str, FeatureGroup]
    data_size: int
    untrained_size: int

    def __init__(
        self,
        model_config: SegmentSumMLPConfig | None = None,
        extractor: FeatureExtractor | None = None,
    ):
        model_config = model_config or SegmentSumMLPConfig()
        extractor = extractor or PerStoreFeature(extract_workload=True)

        self.model = SegmentSumMLP(**model_config.to_dict())
        self.data = OrderedDict()
        self.data_size = 0
        self.untrained_size = 0
        self.extractor = extractor

    def load(  # pylint: disable=too-many-locals
        self,
        path: str,
        target: str = "nvidia/nvidia-v100",
    ) -> None:
        """Load the cached model, cached features, or raw data.

        Parameters
        ----------
        path: str
            The path to the tar file containing cached model, cached features,
            or raw data.
        target: str
            The target for the tuning context.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pth")
            cache_path = os.path.join(tmp_dir, "cached_data.npy")
            raw_path = os.path.join(tmp_dir, "raw_data")
            untar(path, tmp_dir)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
            if os.path.exists(cache_path):
                for group_hash, features, costs in np.load(cache_path, allow_pickle=True):
                    self.data[group_hash] = FeatureGroup(
                        group_hash=group_hash,
                        features=list(features),
                        costs=costs,
                    )
                    self.data_size += len(costs)
                    self.untrained_size += len(costs)
            elif os.path.exists(raw_path):
                from tqdm import tqdm  # type: ignore

                model_dirs = glob.glob(os.path.join(raw_path, "*"))
                workload_paths = []
                for model_dir in model_dirs:
                    json_files = glob.glob(os.path.join(model_dir, "*.json"))
                    for json_file in json_files:
                        if json_file.endswith("_workload.json"):
                            workload_paths.append(json_file)
                for workload_path in tqdm(workload_paths):
                    # pylint: disable=protected-access,broad-exception-caught
                    try:
                        database = JSONDatabase(
                            path_workload=workload_path,
                            path_tuning_record=workload_path.replace(
                                "_workload.json", "_candidates.json"
                            ),
                        )
                    except (ValueError, RuntimeError):
                        continue
                    candidates, results = [], []
                    tuning_records = database.get_all_tuning_records()
                    if len(tuning_records) == 0:
                        continue
                    for record in tuning_records:
                        candidates.append(record.as_measure_candidate())
                        results.append(RunnerResult(run_secs=record.run_secs, error_msg=None))
                    assert len(candidates) == len(results)
                    context = TuneContext(mod=tuning_records[0].workload.mod, target=Target(target))
                    features, mean_costs = extract_features(
                        context, candidates, results, self.extractor
                    )
                    self.add_to_group(features, mean_costs, shash2hex(context.mod))

    def save(self, path: str) -> None:
        """Cache the model and data.

        Parameters
        ----------
        path: str
            The path to the cached tar file.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pth")
            cache_path = os.path.join(tmp_dir, "cached_data.npy")
            torch.save(self.model.state_dict(), model_path)
            data = [
                (
                    g.group_hash,
                    g.features,
                    g.costs,
                )
                for g in self.data.values()
            ]
            np.save(
                file=cache_path,
                arr=np.array(data, dtype=object),
            )
            tar(path, [x for x in [model_path, cache_path] if x is not None])
            logger.info("Saved MLPModel to %s", path)

    def add_to_group(
        self,
        features: list[np.ndarray],
        costs: np.ndarray,
        group_hash: str,
    ):
        """Add features and costs to the data groups with key group_hash.

        Parameters
        ----------
        features: List[np.ndarray]
            The feature vectors.
        costs: np.ndarray
            The measured results.
        group_hash: str
            The structural hash of the candidates.
        """
        group = self.data.get(group_hash, None)
        if group is None:
            group = FeatureGroup(
                group_hash=group_hash,
                features=features,
                costs=costs,
            )
        else:
            group.append(features, costs)
        self.data[group_hash] = group
        self.data_size += len(features)
        self.untrained_size += len(features)


class SegmentSumMLPTrainer:
    """The trainer for Segment Sum MLP model.

    Parameters
    ----------
    state: State
        The state of the trainer.
    batch_size : int
        The batch size.
    learning rate : float
        The learning rate.
    weight decay : float
        The weight decay.
    num_epoch_full : int
        The number of epochs used in full training.
    num_epoch_incremental : int
        The number of epochs used in incremental training.
    grad_clip_norm: float
        The norm of gradient clipping.
    train_verbose: int
        The verbose frequency for training in batches.
    test_interval: int
        The testing interval in epochs.
    test_split: float
        The fraction of data for testing.
    frozen: bool
        Determine whether to re-train the model or not.
    optimizer: "torch.optim.adam.Adam"
        The optimizer.
    scheduler: "torch.optim.lr_scheduler.StepLR"
        The scheduler.
    """

    state: State
    batch_size: int = 128
    learning_rate: float = 7e-4
    weight_decay: float = 1e-6
    num_epoch_full: int = 50
    num_epoch_incremental: int = 5
    grad_clip_norm: float = 0.5
    train_verbose: int = 1000
    test_interval: int = 1
    test_split: float = 0.2
    frozen: bool = False
    optimizer: "torch.optim.adam.Adam"  # type: ignore
    scheduler: "torch.optim.lr_scheduler.StepLR"  # type: ignore

    def __init__(
        self,
        train_config: TrainerConfig | None = None,
        state: State | None = None,
    ):
        train_config = train_config or TrainerConfig()
        state = state or State()

        config = train_config.to_dict()
        for attr in config:
            setattr(self, attr, config[attr])
        self.state = state
        self.device = "cuda" if torch.cuda.device_count() else "cpu"
        self.optimizer, self.scheduler = None, None

    def train_step(
        self,
        data: tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"],
        batch: int = 0,
        train_loss: float | None = None,
    ) -> float:
        """Helper function for training on a single batch.

        Parameters
        ----------
        data: Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]
            A batch of data, should be a tuple of (segment_sizes, features, gt_results).
        batch: int = 0
            The current batch number.
        train_loss: Optional[float] = None
            The previous averaged training loss, None if it is the first batch.

        Returns
        -------
        train_loss: float
            The averaged training loss after the current batch.
        """
        segment_sizes, features, gt_results = (
            data[0].to(self.device),
            data[1].to(self.device),
            data[2].to(self.device),
        )
        self.optimizer.zero_grad()
        pred_results = self.state.model(segment_sizes, features)
        loss = lambda_rank_loss(pred_results, gt_results)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.state.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        loss = loss.detach().cpu()
        train_loss = (
            train_loss * 0.95 + loss.item() * 0.05 if train_loss is not None else loss.item()
        )
        segment_sizes, features, gt_results, pred_results = (
            segment_sizes.detach().cpu(),
            features.detach().cpu(),
            gt_results.detach().cpu(),
            pred_results.detach().cpu(),
        )
        if batch % self.train_verbose == 0:
            logger.info("Batch: %d, train loss: %6f", batch, train_loss)
        return train_loss

    def predict_step(
        self,
        data: tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"],
    ):
        """Helper function for predicting (validating) on a single batch.

        Parameters
        ----------
        data: Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]
            A batch of data, should be a tuple of (segment_sizes, features, gt_results).
            gt_results can be None if it is used for predicting.

        Returns
        -------
        pred_results: np.ndarray
            The predicted results for the current batch.
        test_loss_batch: float
            If used for validation, return the test loss for the current batch.
        test_scores_batch: List[float]
            If used for validation, return the topk scores for the current batch.
        """
        test_loss_batch, test_scores_batch = None, []
        segment_sizes, features = (
            data[0].to(self.device),
            data[1].to(self.device),
        )
        gt_results = data[2]
        pred_results = self.state.model(segment_sizes, features)
        segment_sizes, features, pred_results = (
            segment_sizes.detach().cpu(),
            features.detach().cpu(),
            pred_results.detach().cpu(),
        )
        if gt_results is not None:
            test_loss_batch = lambda_rank_loss(pred_results, gt_results).item()
            for k in [1, 5, 10]:
                test_scores_batch.append(topk_score(pred_results, gt_results, k))
        return pred_results.numpy(), test_loss_batch, test_scores_batch

    def train_full(self):  # pylint: disable=too-many-locals
        """Training on the full dataset."""
        # split into training and testing set
        keys = list(self.state.data.keys())
        test_keys = random.sample(keys, k=math.floor(len(keys) * self.test_split))
        train_data = OrderedDict()
        test_data = OrderedDict()
        for key in keys:
            if key in test_keys:
                test_data[key] = self.state.data[key]
            else:
                train_data[key] = self.state.data[key]
        train_features = list(
            itertools_chain.from_iterable([g.features for g in train_data.values()])
        )
        test_features = list(
            itertools_chain.from_iterable([g.features for g in test_data.values()])
        )
        train_results = np.concatenate([g.min_cost / g.costs for g in train_data.values()])
        test_results = np.concatenate([g.min_cost / g.costs for g in test_data.values()])
        train_loader = SegmentDataLoader(
            train_features, train_results, batch_size=self.batch_size, shuffle=True
        )
        test_loader = SegmentDataLoader(
            test_features, test_results, batch_size=self.batch_size, shuffle=False
        )
        self.optimizer = torch.optim.Adam(
            self.state.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.num_epoch_full // 10,
            gamma=0.8,
            verbose=True,
        )
        self.state.model = self.state.model.to(self.device)
        min_test_loss = 1e10
        logger.info("Training size: %d; Testing size: %d", len(train_loader), len(test_loader))

        model_cache_path = tempfile.NamedTemporaryFile().name  # pylint: disable=consider-using-with
        for epoch in range(self.num_epoch_full):
            logger.info("Epoch: %d", epoch)
            # training
            self.state.model.train()
            train_loss = None
            for batch, data in enumerate(train_loader):
                train_loss = self.train_step(data, batch, train_loss)
            self.scheduler.step()
            # testing
            if epoch % self.test_interval == 0:
                self.state.model.eval()
                test_losses, test_scores = [], []
                for data in test_loader:
                    _, test_loss_batch, test_scores_batch = self.predict_step(data)
                    test_losses.append(test_loss_batch)
                    test_scores.append(test_scores_batch)
                test_loss = (
                    np.array(test_losses[:-1]).mean() if len(test_losses) > 1 else test_losses[0]
                )
                logger.info(
                    "Average test loss: %6f, top1 score: %5f, top5 score: %5f, top10 score: %5f",
                    test_loss,
                    np.array(test_scores)[:, 0].mean(),
                    np.array(test_scores)[:, 1].mean(),
                    np.array(test_scores)[:, 2].mean(),
                )
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    torch.save(self.state.model.state_dict(), model_cache_path)
        self.state.model.to("cpu").load_state_dict(torch.load(model_cache_path))
        self.state.untrained_size = 0

    def train_incremental(
        self,
        features: list[np.ndarray],
        results: np.ndarray,
    ):
        """Training on incremental data.

        Parameters
        ----------
        features: List[np.ndarray]
            The extracted features.
        results: np.ndarray
            The measured results.
        """
        results = np.min(results) / results
        loader = SegmentDataLoader(features, results, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(
            self.state.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.state.model = self.state.model.to(self.device)
        logger.info("Incremental training size: %d", len(loader))
        for epoch in range(self.num_epoch_incremental):
            logger.info("Epoch: %d", epoch)
            self.state.model.train()
            loss = None
            for batch, data in enumerate(loader):
                loss = self.train_step(data, batch, loss)
        self.state.model.to("cpu")
        self.state.untrained_size = max(0, self.state.untrained_size - len(loader))

    def predict_incremental(
        self,
        features: list[np.ndarray],
        results: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predicting (validating) on incremental data.

        Parameters
        ----------
        features: List[np.ndarray]
            The extracted features.
        results: Optional[np.ndarray]
            The measured results, can be None if used for predicting.

        Returns
        -------
        pred_results: np.ndarray
            The predicted results.
        """
        if results is not None:
            results = np.min(results) / results
        loader = SegmentDataLoader(features, results, batch_size=self.batch_size, shuffle=False)
        self.state.model = self.state.model.to(self.device).eval()
        logger.info("Incremental testing size: %d", len(loader))
        pred_results, losses, scores = [], [], []
        for data in loader:
            pred_results_batch, losses_batch, scores_batch = self.predict_step(data)
            pred_results.append(pred_results_batch)
            losses.append(losses_batch)
            scores.append(scores_batch)
        pred_results = np.concatenate(pred_results)
        if results is not None:
            losses = np.array(losses[:-1]).mean() if len(losses) > 1 else losses[0]
            logger.info(
                "Average test loss: %6f, top1 score: %5f, top5 score: %5f, top10 score: %5f",
                losses,
                np.array(scores)[:, 0].mean(),
                np.array(scores)[:, 1].mean(),
                np.array(scores)[:, 2].mean(),
            )
        return pred_results

    def update(
        self,
        features: list[np.ndarray],
        costs: np.ndarray,
        group_hash: str,
    ):
        """Update the dataset and re-train the model if not frozen.

        Parameters
        ----------
        features: List[np.ndarray]
            The extracted features.
        costs: np.ndarray
            The measured results.
        group_hash: str
            The hash of the group.
        """
        self.state.add_to_group(features, costs, group_hash)
        if not self.frozen:
            self.predict_incremental(features, costs)
            if self.state.untrained_size / self.state.data_size > 0.2:
                self.train_full()
            else:
                self.train_incremental(features, costs)


@derived_object
class MLPModel(PyCostModel):
    """Segment Sum MLP Model

    Parameters
    ----------
    trainer: SegmentSumMLPTrainer
        The trainer for the model, handling the training interface.
    """

    trainer: SegmentSumMLPTrainer

    def __init__(
        self,
        *,
        trainer: SegmentSumMLPTrainer | None = None,
    ):
        super().__init__()
        self.trainer = trainer or SegmentSumMLPTrainer()

    def load(self, path: str) -> None:
        """Load the cost model, cached data or raw data from given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        self.trainer.state.load(path)

    def save(self, path: str) -> None:
        """Save the cost model and data to given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        self.trainer.state.save(path)

    def update(
        self,
        context: TuneContext,
        candidates: list[MeasureCandidate],
        results: list[RunnerResult],
    ) -> None:
        """Update the dataset, re-train the cost model if not frozen.

        Parameters
        ----------
        context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.
        results : List[RunnerResult]
            The running results of the measure candidates.
        """
        features, mean_costs = extract_features(
            context, candidates, results, self.trainer.state.extractor
        )
        self.trainer.update(features, mean_costs, shash2hex(context.mod))

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
        features, _ = extract_features(context, candidates, None, self.trainer.state.extractor)
        pred_results = self.trainer.predict_incremental(features)
        return pred_results
