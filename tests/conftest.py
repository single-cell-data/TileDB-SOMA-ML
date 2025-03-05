# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
#
# conftest.py defines pytest fixtures that are available to all test files.
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import pandas as pd
from pytest import fixture
from somacore import AxisQuery
from tiledbsoma import Experiment, Measurement
from tiledbsoma._collection import Collection

from tiledbsoma_ml import ExperimentDataset
from tiledbsoma_ml._common import MiniBatch
from tiledbsoma_ml.dataset import (
    DEFAULT_IO_BATCH_SIZE,
    DEFAULT_OBS_COLUMN_NAMES,
    DEFAULT_SHUFFLE_CHUNK_SIZE,
)

from ._utils import (
    ExpectedBatch,
    XValueGen,
    add_dataframe,
    add_sparse_array,
    assert_batches_equal,
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
obs_column_names = default(DEFAULT_OBS_COLUMN_NAMES)
obs_query = default(None)
# Defaults to True, unless `seed=False` is set (a test-only convenience)
shuffle = default(None)
var_range = default(3)
X_value_gen = default(pytorch_x_value_gen)
shuffle_chunk_size = default(DEFAULT_SHUFFLE_CHUNK_SIZE)
io_batch_size = default(DEFAULT_IO_BATCH_SIZE)
batch_size = default(1)
use_eager_fetch = default(True)
return_sparse_X = default(False)
rank = default(0)
world_size = default(1)
worker_id = default(0)
num_workers = default(1)
seed = default(None)
verify_dataset_shape = default(True)


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
    # `seed=False` is shorthand for `shuffle=False`, used alongside mappings from seed values to expected batch values.
    seed: int | bool | None,
) -> ExperimentDataset:
    """|ExperimentDataset| for testing, constructed from ``soma_experiment`` and other ``fixture`` args.

    ``rank``/``world_size`` control multi-GPU simulation, and ``worker_id``/``num_workers`` simulate multi-worker
    training|DataLoader|'s.
    """
    worker_info = (worker_id, num_workers, seed) if num_workers > 1 else None
    if shuffle is None:
        shuffle = seed is not False
    if seed is False:
        seed = None
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


@fixture
def expected_batches(
    expected: List[List[int]],  # Batches of row idxs
    soma_experiment: Experiment,
    obs_range: int | range,
    var_range: int | range,
    X_value_gen: XValueGen,
    obs_column_names: Sequence[str],
) -> List[ExpectedBatch]:
    """Convert ``expected`` row-idxs to batches of expected data read from ``soma_experiment``.

    ``expected`` contains row indices, which are mapped to ``X`` and ``obs`` objects using the same ``X_value_gen`` /
    ``obs_column_names`` that were used to create the ``soma_experiment``.
    """
    obs = soma_experiment.obs.read().concat().to_pandas()
    if isinstance(obs_range, int):
        obs_range = range(obs_range)
    if isinstance(var_range, int):
        var_range = range(var_range)

    X = X_value_gen(obs_range, var_range)
    X = X.tocsr()
    return [
        (
            [
                X[row_idx + obs_range.start].toarray()[0].tolist()
                for row_idx in row_idx_batch
            ],
            pd.concat(
                [obs.loc[[row_idx], obs_column_names] for row_idx in row_idx_batch]
            ).reset_index(drop=True),
        )
        for row_idx_batch in expected
    ]


@fixture
def check(
    expected_batches: List[ExpectedBatch],
    ds: ExperimentDataset,
    batches: List[MiniBatch],
    var_range: int | range,
    batch_size: int,
    return_sparse_X: bool,
    verify_dataset_shape: bool,
):
    """Verify the ``batches`` produced by ``ds`` match the ``expected`` values.

    Test cases invoke this function by declaring a ``check`` argument. All arguments are themselves ``fixture``s, which
    pytest will evaluate/reuse.
    """
    if verify_dataset_shape:
        assert ds.shape == (len(expected_batches), var_range)
    assert_batches_equal(batches, expected_batches, batch_size, return_sparse_X)
