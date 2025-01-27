# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from functools import partial
from typing import Any, Tuple
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from tiledbsoma import Experiment

from tests._utils import pytorch_x_value_gen
from tiledbsoma_ml import ExperimentDataset
from tiledbsoma_ml.dataloader import experiment_dataloader


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)]
)
def test_multiprocessing__returns_full_result(soma_experiment: Experiment) -> None:
    """Tests that ``ExperimentDataset`` provides all data, as collected from multiple processes
    that are managed by a PyTorch DataLoader with multiple workers configured."""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["soma_joinid", "label"],
            io_batch_size=3,  # two chunks, one per worker
        )
        # Wrap with a DataLoader, which sets up the multiprocessing
        dl = experiment_dataloader(ds, num_workers=2)

        batches = list(iter(dl))

        soma_joinids = np.concatenate(
            [obs["soma_joinid"].to_numpy() for _, obs in batches]
        )
        assert sorted(soma_joinids) == list(range(6))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(3, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_experiment_dataloader__non_batched(
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(ds)
        batches = list(iter(dl))
        assert all(X.shape == (3,) for X, _ in batches)
        assert all(obs.shape == (1, 1) for _, obs in batches)

        X, obs = batches[0]
        assert X.tolist() == [0, 1, 0]
        assert obs["label"].tolist() == ["0"]


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(6, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_experiment_dataloader__batched(
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(ds)
        batches = list(iter(dl))

        X, obs = batches[0]
        assert X.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert obs.to_numpy().tolist() == [[0], [1], [2]]


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [
        (10, 3, pytorch_x_value_gen, use_eager_fetch)
        for use_eager_fetch in (True, False)
    ],
)
def test_experiment_dataloader__batched_length(
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(ds)
        assert len(dl) == len(list(dl))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,batch_size",
    [(10, 3, pytorch_x_value_gen, batch_size) for batch_size in (1, 3, 10)],
)
def test_experiment_dataloader__collate_fn(
    soma_experiment: Experiment,
    batch_size: int,
) -> None:
    def collate_fn(
        batch_size: int, data: Tuple[npt.NDArray[np.number[Any]], pd.DataFrame]
    ) -> Tuple[npt.NDArray[np.number[Any]], pd.DataFrame]:
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

    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            batch_size=batch_size,
            shuffle=False,
        )
        dl = experiment_dataloader(ds, collate_fn=partial(collate_fn, batch_size))
        assert len(list(dl)) > 0


def test_experiment_dataloader__unsupported_params__fails() -> None:
    with patch("tiledbsoma_ml.dataset.ExperimentDataset") as dummy_exp_dataset:
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_dataset, shuffle=True)
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_dataset, batch_size=3)
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_dataset, batch_sampler=[])
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_dataset, sampler=[])
