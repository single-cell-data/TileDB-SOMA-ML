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
    pytorch_x_value_gen,
)


@fixture(scope="function")
def soma_experiment(
    tmp_path: Path,
    obs_range: Union[int, range],
    var_range: Union[int, range],
    X_value_gen: XValueGen,
) -> Experiment:
    """|Experiment| for testing.

    Predictable obs, var, and X values allow tests to verify output of |ExperimentDataset| / |experiment_dataloader|
    operations.
    """
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


# Default-value fixtures, can be overridden on a per-test-case basis using ``parametrize`` / ``params`` / ``sweep``
obs_column_names = default(("soma_joinid",))
obs_query = default(None)
# Defaults to True, unless `seed=False` is set (a test-only convenience)
shuffle = default(None)
var_range = default(3)
X_value_gen = default(pytorch_x_value_gen)
shuffle_chunk_size = default(64)
io_batch_size = default(2**16)
batch_size = default(1)
use_eager_fetch = default(True)
return_sparse_X = default(False)
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
    shuffle: bool | None,
    shuffle_chunk_size: int,
    io_batch_size: int,
    batch_size: int,
    return_sparse_X: bool,
    use_eager_fetch: bool,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
    # `seed=False` is test-only shorthand for `shuffle=False`, used alongside mappings from seed values to expected
    # batch values.
    seed: int | bool | None,
) -> ExperimentDataset:
    """|ExperimentDataset| for testing, constructed from ``soma_experiment`` and other ``fixture`` args.

    ``rank``/``world_size`` control multi-GPU simulation, and ``worker_id``/``num_workers`` simulate multi-worker
    training|DataLoader|'s.
    """
    worker_info = (worker_id, num_workers, seed) if num_workers > 1 else None
    if shuffle is None:
        shuffle = seed is not False
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
        yield ds
