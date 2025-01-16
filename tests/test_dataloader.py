# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
from __future__ import annotations

from typing import Iterator, List, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
from pytest import fixture, raises
from torch.utils.data import DataLoader

from tests._utils import assert_array_equal, param, sweep
from tiledbsoma_ml.common import Batch, NDArrayNumber
from tiledbsoma_ml.dataloader import experiment_dataloader
from tiledbsoma_ml.dataset import ExperimentDataset


@fixture
def dataloader(ds: ExperimentDataset, num_workers: int):
    """Wrap an |ExperimentDataset| fixture in a |DataLoader|, for use in tests."""
    yield experiment_dataloader(ds, num_workers=num_workers)


@fixture
def batch_iter(dataloader: DataLoader) -> Iterator[Batch]:
    """Iterator over a |DataLoader|'s |Batch|'s."""
    return iter(dataloader)


@fixture
def batches(batch_iter: Iterator[Batch]) -> List[Batch]:
    """List of a |DataLoader|'s |Batch|'s."""
    return list(batch_iter)


@param(obs_range=6)
def test_returns_full_result(batches: List[Batch]):
    """Tests that ``ExperimentDataset`` provides all data, as collected from multiple processes that are managed by a
    PyTorch DataLoader with multiple workers configured."""
    soma_joinids = np.concatenate([obs["soma_joinid"].to_numpy() for _, obs in batches])
    assert sorted(soma_joinids) == list(range(6))


@param(obs_range=3, obs_column_names=["label"], shuffle=False)
@sweep(use_eager_fetch=[True, False])
def test_non_batched(batches: List[Batch]):
    assert all(X.shape == (3,) for X, _ in batches)
    assert all(obs.shape == (1, 1) for _, obs in batches)
    X, obs = batches[0]
    assert_array_equal(X, [0, 0.1, 0])
    assert obs["label"].tolist() == ["0"]


@param(obs_range=6, batch_size=3, shuffle=False)
@sweep(use_eager_fetch=[True, False])
def test_batched(batches: List[Batch]):
    X, obs = batches[0]
    assert_array_equal(X, [[0, 0.1, 0], [1, 0, 1.2], [0, 2.1, 0]])
    assert obs.to_numpy().tolist() == [[0], [1], [2]]


@param(obs_range=10, batch_size=3, shuffle=False, obs_column_names=["label"])
@sweep(use_eager_fetch=[True, False])
def test_batched_length(dataloader: DataLoader):
    assert len(dataloader) == len(list(dataloader))


@param(obs_range=10, obs_column_names=["label"], shuffle=False)
@sweep(batch_size=[1, 3, 10])
def test_collate_fn(ds: ExperimentDataset, batch_size: int):
    def collate_fn(
        data: Tuple[NDArrayNumber, pd.DataFrame]
    ) -> Tuple[NDArrayNumber, pd.DataFrame]:
        assert isinstance(data, tuple)
        assert len(data) == 2
        assert isinstance(data[0], np.ndarray) and isinstance(data[1], pd.DataFrame)
        if batch_size > 1:
            assert data[0].shape[0] == data[1].shape[0]
            assert data[0].shape[0] <= batch_size
        else:
            assert data[0].ndim == 1
        assert data[1].shape[1] <= batch_size
        return data

    dl = experiment_dataloader(ds, collate_fn=collate_fn)
    assert len(list(dl)) > 0


def test_unsupported_params_fail():
    with patch("tiledbsoma_ml.dataset.ExperimentDataset") as mock_dataset:
        with raises(ValueError):
            experiment_dataloader(mock_dataset, shuffle=True)
        with raises(ValueError):
            experiment_dataloader(mock_dataset, batch_size=3)
        with raises(ValueError):
            experiment_dataloader(mock_dataset, batch_sampler=[])
        with raises(ValueError):
            experiment_dataloader(mock_dataset, sampler=[])
