# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
#
# conftest.py defines pytest fixtures that are available to all test files.
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

from pytest import fixture
from somacore import AxisQuery
from tiledbsoma import Experiment, Measurement
from tiledbsoma._collection import Collection

from tiledbsoma_ml import ExperimentDataset

from ._utils import (
    XValueGen,
    add_dataframe,
    add_sparse_array,
    default,
    mock_distributed,
    parametrize,
    pytorch_x_value_gen,
)


@fixture(scope="function")
def soma_experiment(
    tmp_path: Path,
    obs_range: Union[int, range],
    var_range: Union[int, range],
    X_value_gen: XValueGen,
) -> Experiment:
    with Experiment.create((tmp_path / "exp").as_posix()) as exp:
        if isinstance(obs_range, int):
            obs_range = range(obs_range)
        if isinstance(var_range, int):
            var_range = range(var_range)

        add_dataframe(exp, "obs", obs_range)
        ms = exp.add_new_collection("ms")
        rna = ms.add_new_collection("RNA", Measurement)
        add_dataframe(rna, "var", var_range)
        rna_x = rna.add_new_collection("X", Collection)
        add_sparse_array(rna_x, "raw", obs_range, var_range, X_value_gen)

    return Experiment.open((tmp_path / "exp").as_posix())


# Default-value fixtures
obs_column_names = default(("soma_joinid",))
obs_query = default(None)
shuffle = default(True)
var_range = default(3)
X_value_gen = default(pytorch_x_value_gen)
shuffle_chunk_size = default(64)
io_batch_size = default(2**16)
batch_size = default(1)
use_eager_fetch = default(True)
return_sparse_X = default(False)
expected_shape = default(None)
rank = default(0)
world_size = default(1)
worker_id = default(0)
num_workers = default(1)
seed = default(None)


@fixture
def ds(
    soma_experiment: Experiment,
    obs_query: AxisQuery | None,
    obs_column_names: Tuple[str],
    shuffle: bool,
    shuffle_chunk_size: int,
    io_batch_size: int,
    batch_size: int,
    return_sparse_X: bool,
    use_eager_fetch: bool,
    expected_shape: Tuple[int, ...] | None,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
    seed: int | None,
) -> ExperimentDataset:
    worker_info = (worker_id, num_workers, seed) if num_workers > 1 else None
    with (
        mock_distributed(rank, world_size, worker_info),
        soma_experiment.axis_query(
            measurement_name="RNA", obs_query=obs_query
        ) as query,
    ):
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=obs_column_names,
            shuffle=shuffle,
            shuffle_chunk_size=shuffle_chunk_size,
            seed=seed,
            io_batch_size=io_batch_size,
            batch_size=batch_size,
            return_sparse_X=return_sparse_X,
            use_eager_fetch=use_eager_fetch,
        )
        if expected_shape is not None:
            assert ds.shape == expected_shape
        yield ds


def rank_sweep(*world_sizes: int):
    return parametrize(
        "rank,world_size",
        [
            (rank, world_size)
            for world_size in world_sizes
            for rank in range(world_size)
        ],
    )


def worker_sweep(num_workers: int):
    return parametrize(
        "worker_id,num_workers",
        [(worker_id, num_workers) for worker_id in range(num_workers)],
    )
